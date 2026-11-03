"""Stage 6B: LangGraph now dispatches actual validated plan steps.

Plan creation, retrieval, routing, tool coordination, and verification are
separate nodes. Mock is the default; a local model adapter is opt-in.
"""
from langgraph.graph import StateGraph, START, END
from .state import EdgeState
from .modules.planner import plan
from .modules.rag import retrieve
from .modules.router import route_model
from .modules.coordinator import run_tools
from .modules.verifier import verify
from .modules.recovery import recover
from .planning.execution import (
    dispatch, record_retrieval, record_tool,
    record_verification, prepare_retry,
)


def route_step(state: EdgeState) -> str:
    op = state.get('active_operation')
    if op is None:
        return 'done'
    if op == 'retrieve_evidence':
        return 'rag'
    if op in {'generate_text', 'generate_code', 'classify_input'}:
        return 'router'
    if op == 'verify_output':
        return 'verifier'
    raise ValueError(f'Unsupported planned operation: {op}')


def after_verification(state: EdgeState) -> str:
    return 'recovery' if state['verification']['status'] in {'FAILED', 'BLOCKED'} else 'done'


def after_recovery(state: EdgeState) -> str:
    return 'retry' if state['recovery_action'] == 'retry' else 'done'


def make_graph():
    builder = StateGraph(EdgeState)
    for name, callback in (
        ('planner', plan),
        ('dispatch', dispatch),
        ('rag', retrieve),
        ('record_retrieval', record_retrieval),
        ('router', route_model),
        ('tools', run_tools),
        ('record_tool', record_tool),
        ('verifier', verify),
        ('record_verification', record_verification),
        ('recovery', recover),
        ('prepare_retry', prepare_retry),
    ):
        builder.add_node(name, callback)

    builder.add_edge(START, 'planner')
    builder.add_edge('planner', 'dispatch')
    builder.add_conditional_edges('dispatch', route_step,
                                  {'rag': 'rag', 'router': 'router',
                                   'verifier': 'verifier', 'done': END})
    builder.add_edge('rag', 'record_retrieval')
    builder.add_edge('record_retrieval', 'dispatch')
    builder.add_edge('router', 'tools')
    builder.add_edge('tools', 'record_tool')
    builder.add_edge('record_tool', 'dispatch')
    builder.add_edge('verifier', 'record_verification')
    builder.add_conditional_edges('record_verification', after_verification,
                                  {'recovery': 'recovery', 'done': END})
    builder.add_conditional_edges('recovery', after_recovery,
                                  {'retry': 'prepare_retry', 'done': END})
    builder.add_edge('prepare_retry', 'dispatch')
    return builder.compile()
