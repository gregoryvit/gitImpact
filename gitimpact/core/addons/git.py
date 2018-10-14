from ..calc.base import Atom, Container, HashContainer
import re
import subprocess
import os


class GitDiffExtractor:
    def __init__(self, repo):
        self.repo = repo

    def process_diff_out(self, diff_out):
        diff_results = diff_out.split('\n')

        cur_in_file = None
        cur_end_file = None
        cur_block = []
        res_blocks = []
        for res in diff_results:
            start_diff_file = re.findall('diff --git.*', res)
            if len(start_diff_file) > 0 and cur_block:
                res_blocks.append({
                    "code": "\n".join(cur_block),
                    "in_file": cur_in_file.strip('a').strip(),
                    "end_file": cur_end_file.strip('b').strip()
                })
                cur_block = []

            in_file_results = re.findall('--- (.*)', res)
            if len(in_file_results) > 0:
                cur_in_file = in_file_results[0]
                continue

            end_file_results = re.findall('\+\+\+ (.*)', res)
            if len(end_file_results) > 0:
                cur_end_file = end_file_results[0]
                continue

            first_line_block_results = re.findall('@@[^@]+@@(.*)', res)
            if len(first_line_block_results) > 0:
                # End current blocks
                cur_line = first_line_block_results[0]
                if cur_block:
                    res_blocks.append({
                        "code": "\n".join(cur_block),
                        "in_file": cur_in_file.strip('a/').strip(),
                        "end_file": cur_end_file.strip('b/').strip()
                    })

                # Start new block
                cur_block = [cur_line]

            cur_line = re.findall('^[+-](.*)', res)
            if len(cur_line) > 0:
                cur_line = re.findall('^[+-](.*)', res)
                if len(cur_line) > 0:
                    cur_line = cur_line[0]
                else:
                    cur_line = ''
                cur_block.append(cur_line)

        if cur_block:
            res_blocks.append({
                "code": "\n".join(cur_block),
                "in_file": cur_in_file.strip('a/').strip(),
                "end_file": cur_end_file.strip('b/').strip()
            })

        return res_blocks

    def diff_versions(self, commit_1=None, commit_2=None):
        try:
            commands = ["git", "diff", "-U0"]
            if commit_1 is not None:
                if commit_2 is None:
                    commit_2_sha = commit_1.hexsha + '~1'
                else:
                    commit_2_sha = commit_2.hexsha
                commit_1_sha = commit_1.hexsha
                commands.extend([commit_1_sha, commit_2_sha])
            out = subprocess.check_output(commands, cwd=self.repo.working_dir)
        except subprocess.CalledProcessError as ex:
            return ex.output

        out_str = out.decode('utf-8')

        return self.process_diff_out(out_str)


class GitDiffAtomBuilder:
    def __init__(self, git_file_atom):
        self.file_path = git_file_atom.file_path
        self.git_stats = git_file_atom.stats

    @property
    def atoms(self):
        return [Atom(self.file_path, self.file_path, [])] * self.git_stats['lines']


class GitStatusAtomBuilder:
    def __init__(self, git_status_atom):
        self.repo = git_status_atom.repo

    @property
    def atoms(self):
        diff_extractor = GitDiffExtractor(self.repo)
        diff_code_blocks = [block for block in diff_extractor.diff_versions()]

        res_atoms = []
        for block in diff_code_blocks:
            file_path = block['end_file']
            lines = block['code'].split('\n')[1:]
            res_atoms.extend([Atom(file_path, file_path, [])] * len(lines))
        return res_atoms


class StringsFileAtomBuilder:
    def __init__(self, git_file_atom):
        self.file_path = git_file_atom.file_path
        self.git_stats = git_file_atom.stats
        self.repo = git_file_atom.commit.repo
        self.commit = git_file_atom.commit
        self.full_path = os.path.join(self.repo.working_dir, self.file_path)

        diff_extractor = GitDiffExtractor(self.repo)
        self.diff_code_blocks = [block for block in diff_extractor.diff_versions(self.commit) if
                                 block['in_file'].endswith(self.file_path)]

    @property
    def atoms(self):
        result_atoms = []
        for code_block in self.diff_code_blocks:
            ids = re.findall(r'^"([^"]+)" *= ', code_block['code'], flags=re.MULTILINE)

            def make_atom(item_id):
                name = "%s@%s" % (self.file_path, item_id)
                return Atom(name, name, [])

            res_atoms = [make_atom(cur_id) for cur_id in ids]
            result_atoms.extend(res_atoms)
        return result_atoms


class GitAtom(Atom):
    def __init__(self, commit):
        self.value = "commit:" + commit.hexsha
        self.commit = commit


class GitFileAtomic(Atom):
    def __init__(self, file_path, stats, commit):
        self.value = "git_file:" + file_path
        self.file_path = file_path
        self.stats = stats
        self.commit = commit


class GitStatusAtom(Atom):
    def __init__(self, repo):
        self.value = "git_status"
        self.repo = repo


class GitUtils:

    def __init__(self):
        pass

    @staticmethod
    def commit_to_container(cur_commit):
        commit_atom = GitAtom(cur_commit)
        builder = AtomsBuilder()
        return builder.process_atoms(commit_atom)

    @staticmethod
    def repo_status_to_container(repo):
        status_atom = GitStatusAtom(repo)
        builder = AtomsBuilder()
        return builder.process_atoms(status_atom)


import concurrent.futures as futures
import functools

class GitRepoLoader:
    def __init__(self, repo):
        self.repo = repo

    def load_commits(self, commit=None, commits_depth=50):
        def filter_commit(cur_commit):
            message = cur_commit.message
            if message.startswith('Merge') or \
                    message.startswith('Version Bump'):
                return False
            return True

        def get_previous_commits(cur_commit, commits_count=50):
            if cur_commit is None:
                rev = "HEAD~1"
            else:
                rev = cur_commit.hexsha + '~1'
            max_count = -1

            if commits_count is None:
                rev = None
            elif commits_count >= 0:
                max_count = commits_count

            return list(self.repo.iter_commits(rev, max_count=max_count))

        previous_commits = get_previous_commits(commit, commits_count=commits_depth)
        previous_commits = filter(filter_commit, previous_commits)

        # previous_commits_containers = list(map(GitUtils.commit_to_container, previous_commits))

        previous_commits_containers = []
        print(len(previous_commits))
        with futures.ThreadPoolExecutor(20) as executor:
            future_to_commit = dict((executor.submit(GitUtils.commit_to_container, c), c) for c in previous_commits)

            for future in futures.as_completed(future_to_commit):
                commit = future_to_commit[future]
                if future.exception() is not None:
                    print('%r generated an exception: %s' % (commit, future.exception()))
                else:
                    previous_commits_containers.append(future.result())

        return previous_commits_containers


class AtomsBuilder:
    def __init__(self, rules=None):
        if rules is None:
            def atoms_builder(builder):
                return lambda a: Container(name=a.value, children=builder(a).atoms)

            def commit(a):
                commit_files = []
                for file_path, stats in a.commit.stats.files.items():
                    commit_files.append(GitFileAtomic(file_path, stats, a.commit))
                res_container = Container(name=a.value, children=commit_files)
                res_container.message = a.commit.message
                return res_container

            self.rules = {
                "^git_status$": atoms_builder(GitStatusAtomBuilder),
                "^commit:.*$": commit,
                "^git_file:.*\.strings$": atoms_builder(StringsFileAtomBuilder),
                "^git_file:.*$": atoms_builder(GitDiffAtomBuilder),
            }
        else:
            self.rules = rules

    def process_atoms(self, atom):
        queue = [(None, atom)]

        result = None

        while queue:
            parent, cur_atom = queue.pop()
            for rule_regex, action in self.rules.items():
                if re.match(rule_regex, cur_atom.value):
                    action_result = action(cur_atom)
                    action_result_type = type(action_result)

                    if action_result_type == Container:
                        queue.extend([(action_result, child) for child in action_result.children])

                    if parent is not None:
                        if cur_atom in parent.children:
                            parent.children.remove(cur_atom)
                            parent.children.append(action_result)
                    else:
                        result = action_result

        return result
