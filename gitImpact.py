# coding=utf-8
__author__ = 'gregoryvit'

import pprint
import os

from git import Repo
from formatters import *

class Task(object):
    def __init__(self, id, format='', regex=''):
        self.raw_id = id
        self.format = format
        self.str_id = format.format(self.raw_id)
        self.regex = regex

    def make(self, id):
        return Task(id, self.format, self.regex)

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
            def get_int_value(value):
                try: 
                    return int(value)
                except:
                    return 0
            commits = {commit_str.split(' ')[0]: {"additions": get_int_value(stats.split('\t')[0]), "deletions": get_int_value(stats.split('\t')[1])} for commit_str, stats in groups}
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


def get_task(string, task_format):
    import re
    m = re.search(task_format, string)
    if m is None:
        return None
    return m.group(0)

def mainGraph(task_id, source_dir, formatters, check_only_child_commits, 
    exclude_task_ids=[], exclude_features=[], out_file_path=None, commits=[], min_weight=0.1, min_impact_rate=0.15, silent=False, limit=None, task_format=('', '')):
    exclude_task_ids.append(task_id)
    original_task = Task(task_id, format=task_format[0], regex=task_format[1])
    git = GitImpactAnalysis(original_task, source_dir)

    all_tasks_count = len(git.get_all_tasks())
    if not silent:
        print "Total tasks: %d" % all_tasks_count

    source_tasks = {}
    commits = {}
    features = {}
    edges = {}

    source_tasks[task_id] = []

    affected_commits = git.get_affected_commits(original_task.str_id) if not commits else commits
    for current_commit in affected_commits:
        commits[current_commit] = []
        print commits

        for file_path in git.get_affected_files(current_commit):
            if file_path in exclude_features:
                continue
            commits_per_file = git.get_commits_per_file(file_path, after=current_commit if check_only_child_commits else None)

            features[file_path] = []

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

            for cur_task in tasks_per_file:
                if cur_task.str_id == original_task.str_id:
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

                if current_commit not in source_tasks[task_id]:
                    source_tasks[task_id].append(current_commit)

                if file_path not in commits[current_commit]:
                    commits[current_commit].append(file_path)

                if cur_task not in features[file_path]:
                    features[file_path].append(cur_task)

    result_edges = [(task, weight) for task, weight in edges.iteritems() if weight > min_weight]
    result_edges = sorted(result_edges, key=lambda x: x[1], reverse=True)
    if limit:
        result_edges = result_edges[:limit]

    return [formatter.format_tasks(source_tasks, commits, features, result_edges) for formatter in formatters]


def main(
    task_id, 
    source_dir, 
    formatters,
    check_only_child_commits, 
    commits=[], 
    min_weight=0.1, 
    min_impact_rate=0.15, 
    silent=False, 
    limit=None, 
    excluded_tasks=[], 
    exclude_features=[],
    output_filepath=None,
    task_format=('', '')):

    return mainGraph(
        task_id, 
        source_dir, 
        formatters, 
        check_only_child_commits, 
        excluded_tasks, 
        exclude_features, 
        commits=commits, 
        min_weight=min_weight, 
        min_impact_rate=min_impact_rate, 
        silent=silent, 
        limit=limit,
        task_format=task_format
    )
