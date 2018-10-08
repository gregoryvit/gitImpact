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
                        "in_file": cur_in_file.strip('a').strip(),
                        "end_file": cur_end_file.strip('b').strip()
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

        return res_blocks

    def diff_versions(self, commit_1, commit_2=None):
        try:
            if commit_2 is None:
                commit_2_sha = commit_1.hexsha + '~1'
            else:
                commit_2_sha = commit_2.hexsha
            commit_1_sha = commit_1.hexsha
            out = subprocess.check_output(["git", "diff", "-U0", commit_1_sha, commit_2_sha], cwd=self.repo.working_dir)
        except subprocess.CalledProcessError as ex:
            return ex.output

        out_str = out.decode('utf-8')

        #         print(out_str)

        return self.process_diff_out(out_str)


class GitDiffAtomBuilder:
    def __init__(self, file_path, git_stats):
        self.file_path = file_path
        self.git_stats = git_stats

    @property
    def atoms(self):
        return [Atom(self.file_path, self.file_path, [])] * self.git_stats['lines']


class StringsFileAtomBuilder:
    def __init__(self, file_path, git_stats, repo, commit):
        self.file_path = file_path
        self.git_stats = git_stats
        self.full_path = os.path.join(repo.working_dir, file_path)

        diff_extractor = GitDiffExtractor(repo)
        self.diff_code_blocks = [block for block in diff_extractor.diff_versions(commit) if
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


class FileAtomsBuilderFabric:
    def __init__(self):
        pass

    @staticmethod
    def get(file_path, stats, repo, commit):
        if file_path.endswith(".strings"):
            return StringsFileAtomBuilder(file_path, stats, repo, commit)
        return GitDiffAtomBuilder(file_path, stats)


class GitUtils:

    def __init__(self):
        pass

    @staticmethod
    def get_file_atoms(file_path, git_stats, cur_repo, cur_commit):
        atoms_builder = FileAtomsBuilderFabric.get(file_path, git_stats, cur_repo, cur_commit)
        if atoms_builder:
            return atoms_builder.atoms
        return None

    @staticmethod
    def commit_to_container(cur_commit, repo):
        result_atomics = []
        for file_path, stats in cur_commit.stats.files.items():
            file_atoms = GitUtils.get_file_atoms(file_path, stats, repo, cur_commit)
            result_atomics.extend(file_atoms)
        container = Container(cur_commit.hexsha, result_atomics)
        container.message = cur_commit.message
        return container


class GitRepoLoader:
    def __init__(self, repo):
        self.repo = repo

    def load_commits(self, commit, commits_depth=50):
        def filter_commit(cur_commit):
            message = cur_commit.message
            if message.startswith('Merge') or \
                    message.startswith('Version Bump'):
                return False
            return True

        def get_previous_commits(cur_commit, commits_count=50):
            return list(self.repo.iter_commits(cur_commit.hexsha + '~1', max_count=commits_count))

        previous_commits = get_previous_commits(commit, commits_count=commits_depth)
        previous_commits = filter(filter_commit, previous_commits)
        previous_commits_containers = list([GitUtils.commit_to_container(c, self.repo) for c in previous_commits])

        return previous_commits_containers
