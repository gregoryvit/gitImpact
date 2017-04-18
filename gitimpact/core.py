# coding=utf-8
__author__ = 'gregoryvit'

from git import Repo, GitCommandError
import re

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

    def __repr__(self):
        return "Task %s" % self.str_id

    def __unicode__(self):
        return unicode(self.__str__())

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

    def get_affected_files_stats(self, commit):
        return self.repo.commit(commit).stats.files

    def get_commits_per_file(self, file_path, after=None):
        try:
            try:
                params = ['--numstat', '--format=oneline']
                if after:
                    params.append(after)
                params.extend(['--follow', file_path])
                message = self.repo.git.log(*params)
            except GitCommandError as e:
                if e.status == 128:
                    params = ['--numstat', '--format=oneline']
                    if after:
                        params.append(after)
                    params.extend(['--', file_path])
                    message = self.repo.git.log(*params)
                else:
                    raise Exception()

            lines = message.split('\n')
            groups = zip(lines[::2], lines[1::2])

            def get_int_value(value):
                try:
                    return int(value)
                except:
                    return 0

            commits = {commit_str.split(' ')[0]: {"additions": get_int_value(stats.split('\t')[0]),
                                                  "deletions": get_int_value(stats.split('\t')[1])} for
                       commit_str, stats in groups}
        except Exception as e:
            # print "%s\n" % e

            commits = {}
        return commits

    def __extract_only_affected_lines(self, source_diff):
        import re
        result_lines = []
        need_save = False
        for line in source_diff.split('\n'):
            if re.match(r'^diff', line, re.MULTILINE):
                need_save = False
            elif re.match(r'^@@', line, re.MULTILINE):
                need_save = True
            elif need_save and re.match(r'^\+|^\-', line, re.MULTILINE):
                result_lines.append(line[1:])

        return "\n".join(result_lines)

    def get_commit_diff(self, commit, file_path):
        commit_obj = self.repo.commit(commit)
        if commit_obj.parents:
            raw_diff = self.repo.git.diff(commit_obj.parents[0].hexsha, commit_obj.hexsha, '-U0', '--', file_path)
        return self.__extract_only_affected_lines(raw_diff)

    def get_total_affections(self, file_path, after=None):
        all_commits_per_file = self.get_commits_per_file(file_path, after=after)
        result_affections = sum(
            [int(value['additions']) + int(value['deletions']) for value in all_commits_per_file.values()])
        return result_affections

    def get_commits_by_diff_contains(self, search_term, after=None):
        try:
            params = ['--numstat', '--format=oneline']
            params.extend(["-S", search_term])
            if after:
                params.append(after)
            message = self.repo.git.log(*params)
            lines = message.split('\n')

            result_commits = {}
            current_commit = None
            current_commit_stats = None
            for line in lines:
                commit_m = re.findall(r'^([0-9a-f]{40})', line, flags=re.MULTILINE)
                if commit_m:
                    current_commit = commit_m[0]
                    result_commits[current_commit] = []
                    current_commit_stats = self.get_affected_files_stats(current_commit)
                    continue
                file_stats_m = re.findall(r'^([0-9]+)\t([0-9]+)\t(.+)', line, flags=re.MULTILINE)
                for additions, deletions, file_path in file_stats_m:
                    if file_path in current_commit_stats:
                        file_stats = current_commit_stats[file_path]
                        result_commits[current_commit].append({
                            "additions": file_stats["insertions"],
                            "deletions": file_stats["deletions"],
                            "file_path": file_path
                        })

            commits = result_commits
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

    def get_last_commit(self, commits):
        if not commits:
            return

        def compare_commits(l_raw_commit, r_raw_commit):
            l_commit, r_commit = self.repo.commit(l_raw_commit), self.repo.commit(r_raw_commit)
            return l_commit.committed_date > r_commit.committed_date

        sorted_commits = sorted(commits, cmp=compare_commits, reverse=True)
        result_commit = sorted_commits[0]
        return result_commit


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


class ImpactGraph(object):
    def __init__(self, debug_out=False):

        #
        # Dict in format:
        #   {
        #       source_task_id: [commits]
        #   }
        # Example:
        #   {
        #       '32604': [
        #           u'74941a9cff89e53fe9abc02fcaf6bcc97a949a39',
        #           u'82f75f333aea7fa0acfb8f8a0e7d63a801bd24f4'
        #       ]
        #   }
        #
        self.source_tasks = {}

        #
        # Dict in format:
        #   {
        #       commit_hex_sha: [
        #           {
        #               'affections': {},
        #               'file_path': "path/to/file"
        #           }
        #       ]
        #   }
        # Example:
        #   {
        #       u'152087f861f45a83589a006075d18b38f94e88dd': [
        #           {
        #               'affections': {
        #                   'deletions': 2,
        #                   'insertions': 0,
        #                   'lines': 2
        #               },
        #               'file_path': u'Path/To/File.txt'}
        #           }
        #       ]
        #   }
        #
        self.commits = {}

        #
        # Dict in format:
        #   {
        #       file_path: [
        #           {
        #               'affections': {},
        #               'commit': "commit_hex_sha",
        #               'total_affections: total_file_affections_int
        #           }
        #       ]
        #   }
        # Example:
        #   {
        #       u'Path/To/File.txt': [
        #           {
        #               'affections': {
        #                   'additions': 18,
        #                   'deletions': 18
        #               },
        #               'commit': u'fd8f24a5055f35013b0962bbfc2ab95776d5e230',
        #               'total_affections': 295
        #           }
        #       ]
        #   }
        #
        self.features = {}

        #
        # Dict in format:
        #   {
        #       file_path: set([
        #           'sub_feature_identifier'
        #       ])
        #   }
        # Example:
        #   {
        #       u'Path/To/File.txt': set([
        #           u'3116A2701E8DA56300A2336F',
        #           u'3116A2711E8DA56300A2336F',
        #           u'3116A2721E8DA62300A2336F'
        #       ])
        #   }
        #
        self.sub_features = {}

        #
        # Dict in format:
        #   {
        #       file_path: set([
        #           'generator_identifier'
        #       ])
        #   }
        # Example:
        #   {
        #       u'Path/To/File.txt': set([
        #           u'3116A2701E8DA56300A2336F',
        #           u'3116A2711E8DA56300A2336F',
        #           u'3116A2721E8DA62300A2336F'
        #       ])
        #   }
        #
        self.generator_ids = {}

        #
        # Dict in format:
        #   {
        #       generator_id: [{
        #           'commit': commit_hex_sha
        #       }]
        #   }
        # Example:
        #   {
        #       u'lalala': [
        #           {
        #               'commit': u'fd8f24a5055f35013b0962bbfc2ab95776d5e230'
        #           }
        #       ]
        #   }
        #
        self.generator_commits = {}

        #
        # Dict in format:
        #   {
        #       commit_hex_sha: [
        #           Task
        #       ]
        #   }
        # Example:
        #   {
        #       u'00ab1c583610755aff0b1d167ef0f5f12a184c07': [Task #1, Task #2]
        #   }
        #
        self.task_commits = {}

        #
        # List of impacted tasks
        # Example:
        #   [Task #1, Task #2]
        #
        self.result_tasks = []

        #
        # List of tasks with weights
        # Example:
        #   [(Task #1, 0.65), (Task #2, 0.32)]
        #
        self.edges = []

        # Cached features total affections
        self.features_total_affections = {}

        # True – enable debug output to terminal
        # False – disable debug output
        self.debug_out = debug_out

    def debug_print(self, source_obj, pretty=False):
        """
        Prints object to console

        :param source_obj: Object that's need to be printed
        :param pretty: Enable formatted output
        """
        if not self.debug_out:
            return
        if pretty:
            import pprint
            pprint.pprint(source_obj)
        else:
            print(source_obj)

    def print_state(self):
        self.debug_print("------------- STATE ----------------")
        self.debug_print("TASKS:")
        self.debug_print(self.source_tasks, pretty=True)
        self.debug_print("\nCOMMITS:")
        self.debug_print(self.commits, pretty=True)
        self.debug_print("\nFEATURES:")
        self.debug_print(self.features, pretty=True)
        self.debug_print("\nSUB FEATURES:")
        self.debug_print(self.sub_features, pretty=True)
        self.debug_print("\nGENERATOR IDS:")
        self.debug_print(self.generator_ids, pretty=True)
        self.debug_print("\nGENERATOR COMMITS:")
        self.debug_print(self.generator_commits, pretty=True)
        self.debug_print("\nTASK COMMITS:")
        self.debug_print(self.task_commits, pretty=True)
        self.debug_print("\nRESULT TASKS:")
        self.debug_print(self.result_tasks)
        self.debug_print("\nEDGES:")
        self.debug_print(self.edges, pretty=True)
        self.debug_print("\n------------------------------------")

    def clean(self):
        # Clean graph

        def check_commit_tasks(commit_tasks_list):
            filtered_tasks = [r_task for r_task in commit_tasks_list if r_task in self.result_tasks]
            return not filtered_tasks

        # Clean commits

        commits_to_delete = [commit for commit, commit_tasks in self.task_commits.iteritems() if
                             check_commit_tasks(commit_tasks)]
        for commit_to_delete in commits_to_delete:
            self.debug_print("DELETE %s" % commit_to_delete)
            del self.task_commits[commit_to_delete]
        for feature in self.features.keys():
            self.features[feature] = [
                commit_dict
                for commit_dict in self.features[feature]
                if commit_dict["commit"] not in commits_to_delete
                ]
        for gen_id in self.generator_commits.keys():
            self.generator_commits[gen_id] = [
                commit_dict
                for commit_dict in self.generator_commits[gen_id]
                if commit_dict["commit"] not in commits_to_delete
                ]

        # Clean features
        features_to_delete = [feature for feature, feature_commits in self.features.iteritems() if not feature_commits]
        for feature_to_delete in features_to_delete:
            self.debug_print("DELETE %s" % feature_to_delete)
            del self.features[feature_to_delete]
        for commit in self.commits.keys():
            self.commits[commit] = [
                concrete_feature
                for concrete_feature in self.commits[commit]
                if concrete_feature["file_path"] not in features_to_delete
                ]

        # Clean input commits
        input_commits_to_delete = [commit for commit, commit_features in self.commits.iteritems() if
                                   not commit_features]
        for input_commit_to_delete in input_commits_to_delete:
            self.debug_print("DELETE %s" % input_commit_to_delete)
            del self.commits[input_commit_to_delete]
        for source_task in self.source_tasks.keys():
            self.source_tasks[source_task] = [
                task_commit
                for task_commit in self.source_tasks[source_task]
                if task_commit not in input_commits_to_delete
                ]

        # Clean sub features

        sub_features_to_delete = [feature for feature in self.sub_features.keys() if feature not in self.features.keys()]
        for sub_feature in sub_features_to_delete:
            del self.sub_features[sub_feature]



# Fulfill

def fulfill_source_tasks(impact_state_graph, git_impact_core, commits=[], original_task=None):
    if commits:
        impact_state_graph.source_tasks = {
            "none": commits
        }
    else:
        impact_state_graph.source_tasks = {
            original_task.raw_id: []
        }

        for current_commit in git_impact_core.get_affected_commits(original_task.str_id):
            if current_commit not in impact_state_graph.source_tasks[original_task.raw_id]:
                impact_state_graph.commits[current_commit] = []
                impact_state_graph.source_tasks[original_task.raw_id].append(current_commit)


def fulfill_commits(impact_state_graph, git_impact_core, exclude_features, sub_features={}, features_generators={}):
    for commit, features_list in impact_state_graph.commits.iteritems():
        result_features = git_impact_core.get_affected_files_stats(commit)
        for file_path, affections in result_features.iteritems():
            if file_path in exclude_features:
                continue

            # If needs filter feature, than extract diff and apply regex to find sub feature
            if file_path in sub_features:
                regex = sub_features[file_path]
                res_diff = git_impact_core.get_commit_diff(commit, file_path)
                l_sub_features = set(re.findall(regex, res_diff))
                if file_path in impact_state_graph.sub_features:
                    impact_state_graph.sub_features[file_path].update(l_sub_features)
                else:
                    impact_state_graph.sub_features[file_path] = l_sub_features

            # Get features generator
            if file_path in features_generators:
                regex = features_generators[file_path]['source_regex']
                res_diff = git_impact_core.get_commit_diff(commit, file_path)
                generator_ids = set(re.findall(regex, res_diff))
                if file_path in impact_state_graph.generator_ids:
                    impact_state_graph.generator_ids[file_path].update(generator_ids)
                else:
                    impact_state_graph.generator_ids[file_path] = generator_ids

            if file_path not in features_list:
                if file_path not in impact_state_graph.features:
                    impact_state_graph.features[file_path] = []
                features_list.append({
                    "file_path": file_path,
                    "affections": affections
                })


def fulfill_features(impact_state_graph, git_impact_core, last_commit=None, check_only_child_commits=True,
                     sub_features={}, features_generators={}):
    if last_commit is None:
        last_commit = git_impact_core.get_last_commit(impact_state_graph.commits.keys())

    generated_commits = {}

    for feature, feature_commits in impact_state_graph.features.iteritems():
        feature_file_path = feature
        commits_per_file = git_impact_core.get_commits_per_file(feature_file_path,
                                                                after=last_commit if check_only_child_commits else None)

        # If needs filter feature, than extract diff and apply regex
        if feature_file_path in impact_state_graph.sub_features and feature_file_path in sub_features:
            result_commits_per_file = {}
            current_sub_features = impact_state_graph.sub_features[feature_file_path]
            regex = sub_features[feature_file_path]

            for commit_per_file, value in commits_per_file.iteritems():
                res_diff = git_impact_core.get_commit_diff(commit_per_file, feature_file_path)
                commit_file_sub_features = set(re.findall(regex, res_diff))
                intersect_features = current_sub_features.intersection(commit_file_sub_features)
                if intersect_features:
                    result_commits_per_file[commit_per_file] = value

            commits_per_file = result_commits_per_file

        # Generate features if needed
        if feature_file_path in impact_state_graph.generator_ids and feature_file_path in features_generators:
            generator_ids = impact_state_graph.generator_ids[feature_file_path]
            feature_generator = features_generators[feature_file_path]

            for file_path_regex, format in feature_generator['dest_files'].iteritems():
                for sub_feature in generator_ids:
                    formatted_str_to_search = format.format(sub_feature)
                    commits_per_diff_search = git_impact_core.get_commits_by_diff_contains(formatted_str_to_search,
                                                                                           after=last_commit if check_only_child_commits else None)
                    impact_state_graph.generator_commits[sub_feature] = [
                        {
                            'commit': commit,
                            'file_path': commit_file_dict['file_path'],
                            'affections': {
                                'additions': commit_file_dict['additions'],
                                'deletions': commit_file_dict['deletions']
                            }
                        }
                        for commit, commit_files in commits_per_diff_search.iteritems() for commit_file_dict in commit_files
                    ]

                    generated_commits.update(commits_per_diff_search)

        # Calc affections
        if feature_file_path not in impact_state_graph.features_total_affections:
            impact_state_graph.features_total_affections[feature_file_path] = sum(
                [int(value['additions']) + int(value['deletions']) for value in commits_per_file.values()])

        total_affections = impact_state_graph.features_total_affections[feature_file_path]

        result_commits = [
            {
                "commit": commit,
                "affections": affections,
                "total_affections": total_affections
            }
            for commit, affections in commits_per_file.iteritems()
            ]

        for feature_commit_dict in result_commits:
            feature_commit = feature_commit_dict["commit"]
            if feature_commit not in feature_commits:
                if feature_commit not in impact_state_graph.task_commits:
                    impact_state_graph.task_commits[feature_commit] = []

                feature_commits.append(feature_commit_dict)

    # Process generated features

    def add_features_to(features_to_add, source_features):
        for adding_feature, commits_dicts in features_to_add.iteritems():
            if adding_feature not in source_features:
                source_features[adding_feature] = commits_dicts
                continue

            source_features[adding_feature].extend(commits_dicts)

    features_to_add = {}
    for commit, feature_affections_dicts in generated_commits.iteritems():
        features_dict = {
            feature_file_path_dict["file_path"]: [{
                'affections': {
                    'additions': feature_file_path_dict['additions'],
                    'deletions': feature_file_path_dict['deletions']
                },
                'commit': commit}]
            for feature_file_path_dict in feature_affections_dicts
        }

        # Add total
        for feature_file_path, commits_dicts in features_dict.iteritems():
            if feature_file_path not in impact_state_graph.features_total_affections:
                impact_state_graph.features_total_affections[feature_file_path] = git_impact_core.get_total_affections(feature_file_path,
                                                                after=last_commit if check_only_child_commits else None)
            total_affections = impact_state_graph.features_total_affections[feature_file_path]
            for commit_dict in commits_dicts:
                commit_dict['total_affections'] = total_affections
        add_features_to(features_dict, features_to_add)

    add_features_to(features_to_add, impact_state_graph.features)
    for commit in generated_commits.keys():
        if commit not in impact_state_graph.task_commits:
            impact_state_graph.task_commits[commit] = []




def fulfill_tasks(impact_state_graph, git_impact_core, exclude_task_ids):
    for commit, commit_tasks in impact_state_graph.task_commits.iteritems():
        tasks_per_commit = filter(
            lambda task: task.raw_id not in exclude_task_ids,
            git_impact_core.get_tasks_from_commit(commit)
        )
        commit_tasks.extend(tasks_per_commit)
        for task in tasks_per_commit:
            if task not in impact_state_graph.result_tasks:
                impact_state_graph.result_tasks.append(task)


# Calculate impact

def calc_impact_in_source_tasks(impact_state_graph):
    source_calculated_tasks = {}

    # Calc

    for task, source_task_commits in impact_state_graph.source_tasks.iteritems():
        result_features = {}
        for commit in source_task_commits:
            source_features = impact_state_graph.commits[commit]
            for feature_dict in source_features:
                feature_file_path = feature_dict["file_path"]
                if feature_file_path in impact_state_graph.features_total_affections:
                    total_affections = impact_state_graph.features_total_affections[feature_file_path]
                    if total_affections != 0:
                        impact = float(feature_dict["affections"]["insertions"] + feature_dict["affections"][
                            "deletions"]) / total_affections
                    else:
                        impact = 0.0
                    feature = feature_dict["file_path"]
                    if feature in result_features:
                        result_features[feature]["impact_level"] += impact
                    else:
                        result_features[feature] = {
                            "impact_level": impact
                        }
        source_calculated_tasks[task] = result_features

    # Merge

    source_files_impact = {}

    for files_impacts in source_calculated_tasks.values():
        for source_file, impact_parameters in files_impacts.iteritems():
            if source_file in source_files_impact:
                source_files_impact[source_file] += impact_parameters["impact_level"]
            else:
                source_files_impact[source_file] = impact_parameters["impact_level"]

    return source_files_impact


def calc_impact_in_result_tasks(impact_state_graph, source_files_impact):
    result_calculated_tasks = {}

    for task in impact_state_graph.result_tasks:
        cur_commits = [commit for commit, commit_tasks in impact_state_graph.task_commits.iteritems() if
                       task in commit_tasks]
        result_features = {}
        for commit in cur_commits:
            for feature, feature_commits in impact_state_graph.features.iteritems():
                f_commits = [f_dict for f_dict in feature_commits if f_dict["commit"] == commit]
                if f_commits:
                    f_commit = f_commits[0]
                    if f_commit["total_affections"] != 0:
                        impact = float(f_commit["affections"]["additions"] + f_commit["affections"]["deletions"]) / \
                                f_commit["total_affections"]
                    else:
                        impact = 1.0
                    if feature in source_files_impact:
                        result_impact = impact / (1.0 - source_files_impact[feature])
                    else:
                        result_impact = impact
                    if feature in result_features:
                        result_features[feature]["impact_level_exclude_source"] += result_impact
                        result_features[feature]["impact_level"] += impact
                    else:
                        result_features[feature] = {
                            "impact_level_exclude_source": result_impact,
                            "impact_level": impact
                        }

        result_calculated_tasks[task] = result_features

    # Merge result tasks features

    merged_result_tasks = {}

    for task, task_features_impacts in result_calculated_tasks.iteritems():
        features_count = len(task_features_impacts)
        features_impact_sum = sum([feature_impact_params["impact_level_exclude_source"] for feature_impact_params in
                                   task_features_impacts.values()])
        merged_result_tasks[task] = features_impact_sum / features_count

    result_edges = sorted(merged_result_tasks.iteritems(), key=lambda x: x[1], reverse=True)
    impact_state_graph.edges = result_edges


# Main

def mainGraph(task_id, source_dir, formatters, check_only_child_commits,
              exclude_task_ids=[], exclude_features=[], out_file_path=None, commits=[], min_weight=0.1,
              min_impact_rate=0.15, silent=False, limit=None, task_format=('', ''), last_commit=None, debug_out=False,
              sub_features={}, features_generators={}):
    def debug_print(items):
        if not debug_out:
            return
        import pprint
        pprint.pprint(items)

    exclude_task_ids.append(task_id)
    original_task = Task(task_id, format=task_format[0], regex=task_format[1])
    git = GitImpactAnalysis(original_task, source_dir)

    all_tasks_count = len(git.get_all_tasks())
    if not silent:
        print "Total tasks: %d" % all_tasks_count

    state = ImpactGraph(debug_out=debug_out)
    state.source_tasks[task_id] = []

    # Fulfill source tasks

    fulfill_source_tasks(state, git, commits=commits, original_task=original_task)
    state.print_state()

    # Fulfill commits

    fulfill_commits(state, git, exclude_features, sub_features=sub_features, features_generators=features_generators)
    state.print_state()

    # Fulfill features

    fulfill_features(state, git, last_commit=last_commit, check_only_child_commits=check_only_child_commits,
                     sub_features=sub_features, features_generators=features_generators)
    state.print_state()

    # Fulfill tasks

    fulfill_tasks(state, git, exclude_task_ids)
    state.print_state()

    # Clean

    state.clean()
    state.print_state()

    # Calculate tasks impact

    # Calc impact in source tasks and merge
    source_files_impact = calc_impact_in_source_tasks(state)
    debug_print(source_files_impact)

    # Calc impact in result tasks and merge
    calc_impact_in_result_tasks(state, source_files_impact)
    debug_print(state.print_state())

    if limit:
        result_edges = state.edges[:limit]
        state.result_tasks = [task for task, _ in result_edges]
        state.clean()
        state.print_state()

    return [formatter.format_tasks(state) for formatter in formatters]


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
        task_format=('', ''),
        debug_out=False,
        sub_features={},
        features_generators={}):
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
        debug_out=debug_out,
        limit=limit,
        task_format=task_format,
        sub_features=sub_features,
        features_generators=features_generators
    )
