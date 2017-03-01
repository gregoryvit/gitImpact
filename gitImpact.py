# coding=utf-8

__author__ = 'gregoryvit'

import pprint
import os

from git import Repo

from redmine import Redmine


class BaseTasksFormatter(object):
    def __init__(self):
        pass

    def prepate_print_data(self, edges):
        print_data = {}

        for idx, edge_info in enumerate(edges):
            task, weight = edge_info
            try:
                task.load_data()

                issue_data = {
                    'task': task,
                    'weight': weight
                }

                version_name = task.version_name if task.version_name is not None else ''
                parent_name = task.parent_name if task.parent_name is not None else ''

                if version_name in print_data:
                    version_data = print_data[version_name]

                    if parent_name in version_data:
                        parent_data = version_data[parent_name]
                        parent_data.append(issue_data)
                        version_data[parent_name] = parent_data
                    else:
                        version_data[parent_name] = [issue_data]

                    print_data[version_name] = version_data
                else:
                    print_data[version_name] = {parent_name: [issue_data]}

                print "%s (%d/%d)" % (task.url, idx + 1, len(edges))
            except Exception as e:
                print "%s (%d/%d) # %s" % (task.url, idx + 1, len(edges), e)

        return print_data

    def format_tasks(self, edges):
        return ""


class FriendlyFormatter(BaseTasksFormatter):
    def __init__(self):
        BaseTasksFormatter.__init__(self)

    def format_tasks(self, edges):
        result_lines = []
        print_data = self.prepate_print_data(edges)

        for version, parents in print_data.iteritems():
            result_lines.append(version)
            for parent, issues in parents.iteritems():
                result_lines.append(u'\t%s' % parent)
                sorted_issues = sorted(issues, key=lambda x: x['weight'], reverse=True)
                for issue in sorted_issues:
                    task = issue['task']
                    result_lines.append(
                        u'\t\t* %s – %s (%f)' % (task.issue_name, task.url, issue['weight']))
        return u'\n'.join(result_lines).encode('utf-8')


class RedmineFormatter(BaseTasksFormatter):
    def __init__(self):
        BaseTasksFormatter.__init__(self)

    def format_tasks(self, edges):
        result_lines = []
        print_data = self.prepate_print_data(edges)

        for version, parents in print_data.iteritems():
            result_lines.append('* %s' % version)
            version_lines = []
            for parent, issues in parents.iteritems():
                for issue in issues:
                    task = issue['task']
                    version_lines.append(
                        (('** "%s":%s – %s (%f)' % (
                            task.str_id, task.url, task.issue_name.encode('utf-8'), issue['weight'])).decode('utf-8'), issue['weight'])
                    )
            version_lines = sorted(version_lines, key=lambda x: x[1], reverse=True)
            for line, weight in version_lines:
                result_lines.append(line)
        return u'\n'.join(result_lines).encode('utf-8')


class BugTrackerIssue():
    def __init__(self, id, host, key):
        self.id = id
        self.redmine_api = Redmine('https://%s' % host, key=key)
        self.parent_name = None
        self.issue_name = None
        self.version_name = None

    def load_data(self):
        self.parent_name = None
        self.issue_name = None
        self.version_name = None

        issue = self.redmine_api.issue.get(self.id)
        self.issue_name = issue.subject

        if issue.parent.id is not None:
            parent_issue = self.redmine_api.issue.get(issue.parent.id)
            self.parent_name = parent_issue.subject
            self.version_name = issue.fixed_version.name


class Task(object):
    def __init__(self, id, format='', regex='', url_format=''):
        self.raw_id = id
        self.str_id = format.format(id)
        self.url = url_format.format(id)
        self.regex = regex

    def make(self, id):
        return Task(id)

    def parse_tasks(self, source_string):
        import re
        re_pattern = re.compile(self.regex)
        match = re_pattern.findall(source_string)
        results = [self.make(result) for result in match]
        return list(set(results))

    def __str__(self):
        return self.str_id

    def __hash__(self):
        return hash(self.raw_id)

    def __eq__(self, other):
        return hash(self.raw_id) == hash(other.raw_id)


class RedmineTask(Task, BugTrackerIssue):
    def __init__(self, id, host):
        Task.__init__(self, id, '#{}', '#([0-9]+)', 'https://%s/issues/{}' % host)
        BugTrackerIssue.__init__(self, id, host, REDMINE_API_KEY)
        self.host = host

    def make(self, id):
        return RedmineTask(id, self.host)


class ImpactAnalysis:
    def __init__(self, task):
        self.task = task

    def get_affected_commits(self, task_id):
        return []

    def get_affected_files(self, commit):
        return []

    def get_commits_per_file(self, file_path, after=None):
        return []

    @staticmethod
    def is_commits_contains(commits):
        def contains_commits(commit):
            return commit not in commits

        return contains_commits

    def get_weights(self, tasks):
        return []

    def merge_task(self, acc, task):
        return []

    def get_tasks_from_commit(self, commit):
        return []

    @staticmethod
    def flat(items):
        return [item for sub_list in items for item in sub_list]

    def get_impacted_tasks(self):
        affected_commits = self.get_affected_commits(self.task.str_id)
        affected_files = map(self.get_affected_files, affected_commits)
        commits_per_files = [filter(lambda i: len(i) > 0, map(self.get_commits_per_file, files)) for files in
                             affected_files]
        u_commits_per_files = filter(lambda i: len(i) > 0,
                                     [filter(self.is_commits_contains(affected_commits), local_commits) for commits in
                                      commits_per_files for local_commits in commits])
        grouped_tasks = [
            self.flat(map(self.get_tasks_from_commit, commits))
            for commits in u_commits_per_files
            ]
        weight_tasks = map(self.get_weights, grouped_tasks)
        merged_tasks = reduce(self.merge_task, self.flat(weight_tasks), {}).items()
        sorted_tasks = sorted(merged_tasks, key=lambda x: x[1], reverse=True)
        return sorted_tasks


class GitImpactAnalysis(object, ImpactAnalysis):
    def __init__(self, task, repo_path):
        ImpactAnalysis.__init__(self, task)
        self.repo = Repo(repo_path)

    def get_all_tasks(self):
        result_tasks = set()
        for commit in self.repo.iter_commits():
            result_tasks.update(self.get_tasks_from_commit(commit))
        return result_tasks

    def get_all_files(self):
        result_files = {}
        for commit in self.repo.iter_commits():
            commit_tasks = self.get_tasks_from_commit(commit)
            commit_files = commit.stats.files.keys()
            for file_path in commit_files:
                if file_path in result_files:
                    result_files[file_path].update(commit_tasks)
                else:
                    result_files[file_path] = set(commit_tasks)

        return result_files

    def get_affected_commits(self, task_id):
        message = self.repo.git.log(format='oneline', grep=task_id)
        commits = map(lambda commit_str: commit_str.split(' ')[0], message.split('\n'))
        return filter(lambda string: len(string) > 0, commits)

    def get_affected_files(self, commit):
        return self.repo.commit(commit).stats.files.keys()

    def get_commits_per_file(self, file_path, after=None):
        try:
            params = ['--numstat', '--format=oneline']
            if after:
                params.append(after)
            params.extend(['--follow', file_path])
            message = self.repo.git.log(*params)
            lines = message.split('\n')
            groups = zip(lines[::2], lines[1::2])
            commits = {commit_str.split(' ')[0]: {"additions": stats.split('\t')[0], "deletions": stats.split('\t')[1]} for commit_str, stats in groups}
        except Exception as e:
            # print "%s\n" % e
            commits = {}
        return commits

    def get_tasks_from_commit(self, commit):
        return self.task.parse_tasks(self.repo.commit(commit).message)

    def get_weights(self, tasks):
        def get_weight(task):
            return task, 1.0 / len(tasks)

        return map(get_weight, tasks)

    def merge_task(self, acc, (key, weight)):
        if key in acc:
            acc[key] = acc[key] + weight if weight < 0.01 else min(acc[key], weight)
        else:
            acc[key] = weight
        return acc


SOURCE_DIR = '/Users/gregoryvit/Development/Swift/surf/NaviAddress-iOS/'


def get_commits(repo):
    return [commit for commit in repo.iter_commits()]


def get_commits_for_file(repo, file_path):
    try:
        message = repo.git.log('--format=oneline', '--follow', file_path)
        commits = map(lambda commit_str: commit_str.split(' ')[0], message.split('\n'))
    except:
        commits = []
    return commits


def map_commits(commits):
    def lol(acc, (key, value)):
        if key in acc:
            acc[key].extend(value)
        else:
            acc[key] = value
        return acc

    def hm(acc, commit):
        current_files = {key: [str(commit)] for key in commit.stats.files.keys()}
        results = reduce(lol, current_files.items(), acc)
        return results

    result = reduce(hm, commits, {})
    return result


def commits_filtered(repo, substring):
    message = repo.git.log(format='oneline', grep=substring)
    commits = map(lambda commit_str: commit_str.split(' ')[0], message.split('\n'))
    return filter(lambda string: len(string) > 0, commits)


TASK_FORMAT = '#[0-9]+'


def get_task(string, task_format):
    import re
    m = re.search(task_format, string)
    if m is None:
        return None
    return m.group(0)


def links(task_ids):
    return map(lambda task_id: REDMINE_BASE_URL + '/issues/' + task_id.replace('#', ''), task_ids)


import graphviz


class StrictDigraph(graphviz.dot.Dot):
    """Directed graph source code in the DOT language."""

    __doc__ += graphviz.Graph.__doc__.partition('.')[2]

    _head = 'strict digraph %s{'
    _edge = '\t\t%s -> %s%s'
    _edge_plain = '\t\t%s -> %s'

def mainGraph(task_id, source_dir, formatter, check_only_child_commits, exclude_task_ids=[], exclude_file_paths=[], out_file_path=None, commits=[], min_weight=0.1, min_impact_rate=0.15):
    exclude_task_ids.append(task_id)
    task = RedmineTask(task_id, '%s' % REDMINE_HOST)
    git = GitImpactAnalysis(task, source_dir)

    dot = StrictDigraph(comment='The Round Table', engine='dot', format='svg')
    dot.graph_attr['rankdir'] = 'LR'

    all_tasks_count = len(git.get_all_tasks())

    print "Total tasks: %d" % all_tasks_count

    edges = {}

    affected_commits = git.get_affected_commits(task_id) if not commits else commits
    for commit in affected_commits:
        # dot.node(commit, commit)
        for file_path in git.get_affected_files(commit):
            if file_path in exclude_file_paths:
                continue
            commits_per_file = git.get_commits_per_file(file_path, after=commit if check_only_child_commits else None)

            total_affections = sum([int(value['additions']) + int(value['deletions']) for value in commits_per_file.values()])

            edits_per_file = [item for 
                sublist in [
                        map(lambda task: (task, int(affections['additions']) + int(affections['deletions'])),git.get_tasks_from_commit(commit))
                        for commit, affections in commits_per_file.iteritems()
                ] for item in sublist if item[0].raw_id not in exclude_task_ids]

            def append_task_item(acc, item):
                task, affections = item
                if task not in acc:
                    acc[task] = {'count': 0, 'affections': 0}

                acc[task] = {'count': acc[task]['count'] + 1, 'affections': acc[task]['affections'] + affections}
                return acc
            tasks = reduce(append_task_item, edits_per_file, {})
            tasks_per_file = tasks.keys()

            if not tasks_per_file or float(len(tasks_per_file)) / float(all_tasks_count) > min_impact_rate:
                # если кол-во затронутых тасков для файла больше минимального значения от общего числа тасков, то пропускаем
                continue

            dot.node(file_path, file_path)
            # dot.edge(commit, file_path)
            dot.edge(task.str_id, file_path)

            for cur_task in tasks_per_file:
                if cur_task.str_id == task.str_id:
                    continue
                cur_task_id = cur_task.str_id

                task_file_edits = tasks[cur_task]['count']

                affections_weight = float(tasks[cur_task]['affections']) / float(total_affections)
                file_weight = task_file_edits / float(len(edits_per_file))
                file_weight = file_weight + affections_weight * 0.5

                if cur_task in edges:
                    edges_count = edges[cur_task] + file_weight
                else:
                    edges_count = file_weight
                edges[cur_task] = edges_count

                dot.node(cur_task_id, "%s (%f)" % (cur_task_id, edges_count), URL=cur_task.url)
                dot.edge(file_path, cur_task_id)

    dot.render('./test.gv', view=False)
    result_edges = [(task, weight) for task, weight in edges.iteritems() if weight > min_weight]
    formatted_result = formatter.format_tasks(result_edges)
    print formatted_result

    if out_file_path is not None:
        with open(out_file_path, "w+") as f:
            f.write(formatted_result)


def main(task_id, source_dir, check_only_child_commits, commits=[], min_weight=0.1, min_impact_rate=0.15):
    excluded_tasks = [
        "28025",
        "27573",
        "28290",
        "26432",
        "29883",
        "25792",
        "26334",
        "35",
        "29129",
        "28739",
        "24609",
        "27663"
    ]

    formatter = RedmineFormatter()
    output_filepath = "test.txt"
    exclude_file_paths = [
        "NaviAddress/Resources/twine.txt",
        "NaviAddress/Resources/en.lproj/Localizable.strings",
        "NaviAddress/Library/Service Layer/Constants.swift",
        "Podfile",
        "Podfile.lock"
    ]
    mainGraph(task_id, source_dir, formatter, check_only_child_commits, excluded_tasks, exclude_file_paths, commits=commits, min_weight=min_weight, min_impact_rate=min_impact_rate)
    return

    task = RedmineTask(task_id, REDMINE_HOST)

    git = GitImpactAnalysis(task, source_dir)

    tasks = git.get_impacted_tasks()
    mapped_tasks = [(task.url, weight) for (task, weight) in tasks]

    for task_data in mapped_tasks:
        print '%s | %s' % task_data
    print '---------------------------------'
