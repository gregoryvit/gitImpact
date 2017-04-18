from BaseFormatter import *

import graphviz


class StrictDigraph(graphviz.dot.Dot):
    """Directed graph source code in the DOT language."""

    __doc__ += graphviz.Graph.__doc__.partition('.')[2]

    _head = 'strict digraph %s{'
    _edge = '\t\t%s -> %s%s'
    _edge_plain = '\t\t%s -> %s'


class GraphvizFormatter(BaseFormatter):
    def __init__(self, silent, output_filepath, output_format):
        BaseFormatter.__init__(self, silent)
        self.output_filepath = output_filepath
        self.output_format = output_format

    def format_tasks(self, graph):
        dot = StrictDigraph(comment='', engine='dot', format=self.output_format)
        dot.graph_attr['rankdir'] = 'LR'
        # dot.graph_attr['overlap'] = 'false'

        for feature, commits_dicts in graph.features.iteritems():
            tasks = [graph.task_commits[commits_dict["commit"]] for commits_dict in commits_dicts]
            flatten_tasks = [task for sub_list in tasks for task in sub_list]
            for task in flatten_tasks:
                dot.edge(feature, task.str_id)

        for commit, commit_features in graph.commits.iteritems():
            for feature_dict in commit_features:
                dot.edge(commit, feature_dict["file_path"], weight='5')

        for task, task_commits in graph.source_tasks.iteritems():
            for commit in task_commits:
                dot.edge(task, commit)

        for feature, sub_features in graph.sub_features.iteritems():
            for sub_feature in sub_features:
                sub_graph = StrictDigraph('subgraph')
                sub_graph.graph_attr.update(rank='same')
                sub_feature_id = "sub_%s" % sub_feature
                sub_graph.node(sub_feature_id, sub_feature, shape='plaintext')
                sub_graph.edge(sub_feature_id, feature)
                dot.subgraph(sub_graph)

        for generator_feature, gen_ids in graph.generator_ids.iteritems():
            for gen_id in gen_ids:
                dot.node(gen_id, shape='box')
                dot.edge(generator_feature, gen_id)

        for gen_id, gen_files in graph.generator_commits.iteritems():
            for gen_file in gen_files:
                dot.edge(gen_id, gen_file['file_path'], style='dotted')

        for task, weight in graph.edges:
            dot.node(task.str_id, "%s (%f)" % (task.str_id, weight))


        dot.render(self.output_filepath, view=False)

        return ""
