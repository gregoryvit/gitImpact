# coding=utf-8

import core

CONFIG_FILENAME = ".gitimpact.yml"
CONFIG_TEMPLATE_FILE_PATH = "templates/init_template.yml"


def create_config_file(directory_path):
    import os
    import shutil
    template_yml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG_TEMPLATE_FILE_PATH)
    result_yml_path = os.path.join(directory_path, CONFIG_FILENAME)
    shutil.copy(template_yml_path, result_yml_path)


def parse_yaml(path_to_yaml_file):
    import ruamel.yaml

    with open(path_to_yaml_file, 'r') as yaml_file:
        try:
            return ruamel.yaml.load(yaml_file, Loader=ruamel.yaml.Loader)
        except ruamel.yaml.YAMLError as exc:
            print(exc)


def get_option(option, config, alt=None, default=None):
    if alt is not None and alt != default:
        return alt

    if option in config:
        return config[option]

    return default


def get_formatters(formatter_dict, silent):
    def get_redmine_generator(key):
        if "issues_generator" in formatter_dict[key]:
            from issues.redmine import RedmineIssuesGenerator

            generator_options = formatter_dict[key]["issues_generator"]
            return RedmineIssuesGenerator(generator_options["host"], generator_options["api_key"])

    result_formatters = []
    for formatter_key, formatter_options in formatter_dict.iteritems():
        if formatter_key == "graphviz_formatter":
            result_formatters.append(core.GraphvizFormatter(silent, **formatter_options))
        elif formatter_key == "redmine_formatter":
            formatter_options["issues_generator"] = get_redmine_generator(formatter_key)
            result_formatters.append(core.RedmineFormatter(silent, **formatter_options))
        elif formatter_key == "friendly_formatter":
            formatter_options["issues_generator"] = get_redmine_generator(formatter_key)
            result_formatters.append(core.FriendlyFormatter(silent, **formatter_options))
    return result_formatters


def main(app_options):
    try:
        yaml_config = parse_yaml(app_options.yaml_config_file)
    except IOError as e:
        print "Configuration file not found. Run `gitimpact init` to create it"
        exit(1)

    task_format = get_option("task_format", yaml_config, default={})

    result_options = {
        "excluded_tasks": get_option("excluded_tasks", yaml_config, default=[]),
        "exclude_features": get_option("exclude_features", yaml_config, default=[]),
        "output_filepath": get_option("output_filepath", yaml_config),
        "task_id": get_option("task_id", yaml_config, alt=app_options.task_id),
        "commits": get_option("commits", yaml_config, alt=app_options.commit, default=[]),
        "min_weight": float(get_option("min_weight", yaml_config, alt=app_options.min_weight, default=0.1)),
        "min_impact_rate": get_option("min_impact_rate", yaml_config, alt=app_options.min_impact_rate, default=0.15),
        "silent": get_option("silent", yaml_config, alt=app_options.silent, default=False),
        "source_dir": app_options.repo_path,
        "sub_features": get_option("sub_features", yaml_config, default={}),
        "debug_out": get_option("debug", yaml_config, alt=app_options.debug, default=False),
        "task_format": (
            get_option("output_format", task_format, default=""), get_option("parse_regex", task_format, default=""))
    }

    check_all_commits = get_option("check_all_commits", yaml_config, alt=app_options.check_all_commits, default=False)
    result_options["check_only_child_commits"] = (not bool(check_all_commits))
    result_options["formatters"] = get_formatters(get_option("formatter", yaml_config, default={}),
                                                  result_options["silent"])

    limit = get_option("limit", yaml_config, alt=app_options.limit)
    if limit:
        result_options["limit"] = int(limit)

    return core.main(**result_options)
