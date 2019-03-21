from graphviz import Digraph
import pydot

def plot_graph(top_var, fname, params=None):
    """
    This method don't support release v0.1.12 caused by a bug fixed in: https://github.com/pytorch/pytorch/pull/1016
    So if you want to use `plot_graph`, you have to build from master branch or wait for next release.
    Plot the graph. Make sure that require_grad=True and volatile=False
    :param top_var: network output Varibale
    :param fname: file name
    :param params: dict of (name, Variable) to add names to node that
    :return: png filename
    """
    dot = Digraph(comment='LRP',
                  node_attr={'style': 'filled', 'shape': 'box'})
    # , 'fillcolor': 'lightblue'})

    seen = set()

    if params is not None:
        assert isinstance(params.values()[0], Variable)
        param_map = {id(v): k for k, v in params.items()}

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = '{}\n '.format(param_map[id(u)]) if params is not None else ''
                node_name = '{}{}'.format(name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    add_nodes(top_var.grad_fn)
    dot.save(fname)
    (graph,) = pydot.graph_from_dot_file(fname)
    im_name = '{}.png'.format(fname)
    #graph.write_png(im_name)
    return im_name

