import json
from pathlib import Path

from loguru import logger as log
import networkx as nx
import torch
from transformers import AutoTokenizer, RobertaModel


def get_edge_index(G, num_nodes, id_map, label):
    src_list = []
    dst_list = []
    for u, v, k in G.edges(keys=True):
        if k == label:
            try:
                src = id_map[u]
                dst = id_map[v]
            except KeyError:
                log.warning(f"edge ({u}, {v}) has no ID mapping")
                continue
            if src < num_nodes and dst < num_nodes:
                src_list.append(src)
                dst_list.append(dst)
            else:
                log.warning(f"edge ({src}, {dst}) can't be in graph with {num_nodes} nodes.")
    return [src_list, dst_list]


def get_extra_info(nodes):
    joern_ids = []
    line_numbers = []
    for node in nodes:
        joern_ids.append(node["id"])
        line_numbers.append(node.get("lineNumber", -1))
    return [joern_ids, line_numbers]


def extract_node_data(nodes, tokenizer: AutoTokenizer):
    subtokens = []
    id_map = {}
    for i, node in enumerate(nodes):
        code = node["code"]
        _subtokens = tokenizer(code, return_tensors="pt")["input_ids"][0]
        subtokens.append(_subtokens)
        id_map[node["id"]] = i
    token_counts = torch.tensor([len(sub) for sub in subtokens], dtype=torch.long)
    token_idx = torch.repeat_interleave(torch.arange(len(nodes)), token_counts)
    x = list(range(len(nodes)))
    subtokens = torch.cat(subtokens)
    return x, subtokens, id_map, token_idx


def graph_to_pyg(sample, base_model: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    pyg_path = Path(sample["pyg_path"])
    if not pyg_path.exists():
        return empty_graph()
    with open(pyg_path, "r") as f:
        graph = json.load(f)[0]

    nodes = graph["nodes"]

    G = nx.MultiDiGraph()
    for node in nodes:
        for dfg_type in ("dfg_c", "dfg_r", "dfg_w"):
            if dfg_type in node:
                G.add_edge(node[dfg_type], node["id"], dfg_type)
        G.add_node(node["id"], label=node["label"], code=node["code"])
    for label in ("ast", "cfg", "dfg"):
        for edge in graph[label]:
            G.add_edge(edge["in"], edge["out"], label)

    x, subtokens, id_map, token_idx = extract_node_data(nodes, tokenizer)
    x_types = [node["label"] for node in nodes]
    joern_ids, line_numbers = get_extra_info(nodes)

    features = dict(
        x=x,
        x_types=x_types,
        token_idx=token_idx,
        subtokens=subtokens,
        ast_edge_index=get_edge_index(G, len(x), id_map, "ast"),
        cfg_edge_index=get_edge_index(G, len(x), id_map, "cfg"),
        dfg_edge_index=get_edge_index(G, len(x), id_map, "dfg"),
        name=graph["name"],
        joern_ids=joern_ids,
        line_numbers=line_numbers,
    )
    return features


def empty_graph():
    return {
        "x": [],
        "x_types": [],
        "token_idx": torch.tensor([], dtype=torch.long),
        "subtokens": torch.tensor([], dtype=torch.long),
        "ast_edge_index": [[], []],
        "cfg_edge_index": [[], []],
        "dfg_edge_index": [[], []],
        "name": "",
        "joern_ids": [],
        "line_numbers": [],
    }


def unpack_and_pad(tokens, token_idx, pad_token, max_per_node=64):
    tokens = torch.tensor(tokens)
    token_idx = torch.tensor(token_idx)
    token_counts = torch.bincount(token_idx)
    out = torch.full((len(token_counts), min(max_per_node, max(token_counts))), pad_token)
    for i, count in enumerate(token_counts):
        out[i, :count] = tokens[token_idx == i][:out.shape[1]]
    return out


@torch.no_grad()
def embed_graph_batch(samples, base_model: str):
    model = RobertaModel.from_pretrained(base_model, add_pooling_layer=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    embeddings = []
    clear_every = 100
    for i, (tokens, token_idx) in enumerate(zip(samples["subtokens"], samples["token_idx"])):
        tokens = unpack_and_pad(tokens, token_idx, model.config.pad_token_id)
        mask = 1 - torch.eq(tokens, model.config.pad_token_id).long()
        emb = model(input_ids=tokens.to(device), attention_mask=mask.to(device)).last_hidden_state[:, 0, :]
        embeddings.append(emb.cpu())
        del tokens, mask, emb
        if i % clear_every == 0:
            torch.cuda.empty_cache()
    return {"x_emb": embeddings}
