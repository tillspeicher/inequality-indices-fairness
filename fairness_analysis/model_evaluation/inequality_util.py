import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from sklearn import metrics
from itertools import chain, combinations
from collections import defaultdict
sys.path.insert(0, '../shared')
import indices
sys.path.insert(0, '../../util')
import output

FIG_TYPE_LORENZ = 'lorenz_curves'
FIG_TYPE_ACC_FAIR = 'accuracy_fairness'
FIG_TYPE_INEQ_DECOMP = 'inequality_decomposition'
FIG_TYPE_CONSTRAINT_INEQ_DECOMP = 'constraint_ineq_decomp'
FIG_TYPE_INTERGROUP_INEQ = 'intergroup_ineq'
FIG_TYPE_INTERGROUP_SPLITS = 'intergroup_splits'

figures = {FIG_TYPE_LORENZ: {}, FIG_TYPE_ACC_FAIR: {}, FIG_TYPE_INEQ_DECOMP: {},
        FIG_TYPE_CONSTRAINT_INEQ_DECOMP: {}}


def clear_figures():
    for fig_type_figures in figures.values():
        fig_type_figures.clear()

def compute_lorenz_curve(diffs):
    diffs.sort()
    diff_sum = abs(np.sum(diffs))
    cdf = np.cumsum(diffs)
    #cdf = cdf / diff_sum
    return cdf

def plot_lorenz_curve(cdf, method_name, fig_name):
    num_diffs = len(cdf)
    x_vals = [i * 1/(num_diffs-1) for i in range(num_diffs)]

    fig = figures[FIG_TYPE_LORENZ].setdefault(
            fig_name, plt.figure())
    ax = fig.gca()
    ax.plot(x_vals, cdf, label=method_name)

def compute_rejection_curves(probas, labels, groups, metrics):
    order = np.argsort(probas)
    labels = labels[order]
    groups = {g_name: group[order] for g_name, group in groups.items()}

    pred_labels = np.ones(len(labels))
    metrics_res = {metric: [metric_computation(labels, pred_labels, groups)] \
            for metric, metric_computation in metrics.items()}
    for i in range(len(probas)):
        pred_labels[i] = 0

        for metric, metric_computation in metrics.items():
            metrics_res[metric].append(
                metric_computation(labels, pred_labels, groups))

    #assert math.isclose(fairnesses[0], fairnesses[-1])

    return metrics_res

def plot_rejection_curves(rejection_curves,
        method_name, fig_name, color=None):
    assert(len(rejection_curves) == 2)

    fig_comb = figures[FIG_TYPE_ACC_FAIR].get(fig_name)
    if fig_comb is None:
        fig = plt.figure()
        ax1 = fig.gca()
        ax2 = ax1.twinx()
        figures[FIG_TYPE_ACC_FAIR][fig_name] = (fig, ax1, ax2)
    else:
        (fig, ax1, ax2) = fig_comb

    num_vals = len(rejection_curves['Accuracy'] if 'Accuracy' in \
            rejection_curves else rejection_curves['Overall'])
    x_vals = [i / num_vals for i in range(num_vals + 1)]

    # subsample
    x_vals = subsample(x_vals)

    for metric, metric_res in rejection_curves.items():
        metric_res = subsample(metric_res)
        if metric == 'Accuracy' or metric == 'Overall':
            ax = ax2
            linestyle = ':'
        else:
            ax = ax1
            linestyle = '-'

        if metric == 'Unfairness':
            ax.set_ylabel("Unfairness ($\mathcal{E}^2$)")
        if metric == 'Overall':
            ax.set_ylabel("Individual unfairness ($\mathcal{E}^2$)\n(dotted lines)")
        elif metric == 'Between-group':
            metric = 'Between'
            ax.set_ylabel("Between-group\nunfairness ($\mathcal{E}_{\\beta}^2$) (solid lines)")
        else:
            ax.set_ylabel(metric)

        ax.plot(x_vals, metric_res, c=color, linestyle=linestyle,
                label="{} ({})".format(method_name, metric))


def get_inequality_decomp(benefits, groups):
        group_names = []
        group_benefits = []
        for g_name, group in groups.items():
            group_benefit = benefits[group]
            group_names.append(g_name)
            group_benefits.append(group_benefit)

        decomp = indices.decompose_inequality(
                group_benefits)
        decomp['group_names'] = group_names
        return decomp

def plot_inequality_decomposition(probas, labels, groups, fig_name):
    order = np.argsort(probas)
    labels = labels[order]
    groups = {g_name: group[order] for g_name, group in groups.items()}

    pred_labels = np.ones(len(labels))
    overall_inequalities = []
    group_inequalities = []
    group_fairnesses = {g_name: ([], [], []) for g_name in groups}
    accuracies = []

    for i in chain([-1], range(len(probas))):
        if i >= 0:
            pred_labels[i] = 0
        benefits = indices.compute_benefits(pred_labels, labels)

        accuracy = metrics.accuracy_score(labels, pred_labels)
        accuracies.append(accuracy)

        decomp = get_inequality_decomp(benefits, groups)
        overall_inequalities.append(decomp['overall_ineq'])
        group_inequalities.append(decomp['intergroup_ineq'])

        for g_name, intragroup_inequality, \
                unweighted_intragroup_inequality, \
                intergroup_component in \
                zip(decomp['group_names'],
                decomp['subgroup_ineqs'],
                decomp['unweighted_ineqs'],
                decomp['intergroup_components']):
            group_fairnesses[g_name][0].append(
                    intragroup_inequality)
            group_fairnesses[g_name][1].append(
                unweighted_intragroup_inequality)
            group_fairnesses[g_name][2].append(
                intergroup_component)

    fig = figures[FIG_TYPE_INEQ_DECOMP].setdefault(
            fig_name, plt.figure())
    ax = fig.gca()
    fig_unweighted = figures[FIG_TYPE_INEQ_DECOMP].setdefault(
            fig_name + "_unweighted", plt.figure())
    ax_unweighted = fig_unweighted.gca()

    x_vals = [i / len(probas) for i in range(len(probas) + 1)]

    # subsampling
    x_vals = subsample(x_vals)
    overall_inequalities = subsample(overall_inequalities)
    group_inequalities_plot = subsample(group_inequalities)

    ax.plot(x_vals, overall_inequalities, linestyle='-',
            label="Overall inequality")
    for axis in [ax, ax_unweighted]:
        axis.plot(x_vals, group_inequalities_plot, linestyle='-',
                label="Between-group inequality")

    for g_name, (intragroup_inequality, unweighted_intragroup_inequality, intergroup_component) in group_fairnesses.items():

        intragroup_inequality = subsample(intragroup_inequality)
        unweighted_intragroup_inequality = subsample(unweighted_intragroup_inequality)
        intergroup_component = subsample(intergroup_component)

        ax.plot(x_vals, intragroup_inequality, linestyle='-',
                label="{} within-group inequality".format(g_name))
        
        #for axis in [ax, ax_unweighted]:
        for axis in [ax_unweighted]:
            axis.plot(x_vals, intergroup_component,
                    linestyle='-',
                    label="{} between-group component".format(g_name))
        ax_unweighted.plot(x_vals,
                unweighted_intragroup_inequality, linestyle='-',
                label="{} within-group inequality".format(g_name))

    return {'intergroup_ineq': group_inequalities,
            'accuracy': accuracies}
    

def plot_curves(fig_type, fig_name, x_vals, x_label,
        curves1, ax_label1,
        curves2=None, ax_label2=None,
        bars=False, colors=None, linestyles=None):
    if fig_type not in figures:
        figures[fig_type] = {}
    assert fig_name not in figures[fig_type], "Figure '{}' already exists".format(fig_name)

    error_config = {'ecolor': 'red', 'capsize': 5, 'capthick': 2}

    assert curves2 is None or not bars

    fig = plt.figure()
    ax = fig.gca()

    if isinstance(x_vals[0], str):
        tick_locs = np.arange(len(x_vals))
        ax.set_xticks(tick_locs)
        ax.set_xticklabels(x_vals)#, rotation=45)
        x_vals = tick_locs

    if bars:
        barwidth = .7 / len(curves1)
        bar_offset = (len(curves1) - 1) * barwidth / 2

    def get_color_linestyle(curve_name):
        if isinstance(curve_name, tuple):
            c1, c2 = curve_name
            color = colors[c1]
            linestyle = linestyles[c2]
            return c1 + " " + c2, color, linestyle

        if colors is not None:
            color = colors[curve_name]
        else:
            color = None
        if linestyles is not None:
            linestyle = linestyles[curve_name]
        else:
            linestyle = None

        return curve_name, color, linestyle

    for i, (curve_name, curve_vals) in enumerate(curves1.items()):
        curve_name, color, linestyle = get_color_linestyle(curve_name)
        conf_intervals = None
        if isinstance(curve_vals, tuple):
            curve_vals, conf_intervals = curve_vals
        if bars:
            ax.bar(x_vals + i * barwidth - bar_offset, curve_vals,
                    barwidth, label=curve_name,
                    color=color, linestyle=linestyle,
                    yerr=conf_intervals, error_kw=error_config)
        else:
            ax.plot(x_vals, curve_vals, label=curve_name,
                    color=color, linestyle=linestyle)
                    #yerr=conf_intervals, error_kw=error_config)

    ax.set_xlabel(x_label)
    ax.set_ylabel(ax_label1)

    if curves2 is not None:
        assert ax_label2 is not None
        ax2 = ax.twinx()
        for curve_name, curve_vals in curves2.items():
            curve_name, color, linestyle = get_color_linestyle(curve_name)
            ax2.plot(x_vals, curve_vals, label=curve_name,
                    color=color, linestyle=linestyle)
        ax2.set_ylabel(ax_label2)

        figures[fig_type][fig_name] = (fig, ax, ax2)
        
    else:
        figures[fig_type][fig_name] = fig

def plot_pie(fig_type, fig_name, sizes, labels):
    assert fig_name not in figures[fig_type], "Figure '{}' already exists".format(fig_name)

    fig = plt.figure()
    ax = fig.gca()
    ax.pie(sizes, labels=labels)

    figures[fig_type][fig_name] = fig


def subsample(curve, target=500, min_points=100):
    if target >= len(curve) or len(curve) <= min_points:
        return curve

    if target > 1:
        frac = target / len(curve)
    else:
        frac = target

    sampling_dist = round(1 / frac)
    subsampled_curve = list(curve[::sampling_dist])
    if len(curve) % sampling_dist != 1:
        subsampled_curve.append(curve[-1])
    assert len(subsampled_curve) >= min_points
    return subsampled_curve

 
def get_figure_name(dataset, fig_type, fig_name):
    return '{}_{}_{}'.format(dataset, fig_type, fig_name)

def get_web_target_dir(out_dir):
    return out_dir.replace('results',
            'inequality_indices/measuring_inequality')

def plot_results(out_dir, dataset, output_channel='upload'):
    for fig_type_name, fig_type_figures in figures.items():
        for fig_name, fig in fig_type_figures.items():
            if isinstance(fig, tuple):
                fig, ax, ax2 = fig
                #ax = fig.gca()
            else:
                ax = fig.gca()
                ax2 = None

            h_dim_pdf = 12
            separate_legend = ax2 is not None

            if fig_type_name == FIG_TYPE_LORENZ:
                ax.plot([0, 1], [0, 1], linestyle='--', label='Equality') # line of equality
                ax.set_xlabel("Fraction of population")
                ax.set_ylabel("Cumulative fraction of loss")
            elif fig_type_name == FIG_TYPE_ACC_FAIR:
                ax.set_xlabel("Fraction of rejected users ($\\tau$)")
                separate_legend = True
                if ax.get_ylabel().startswith('Between'):
                    h_dim_pdf = 14
            elif fig_type_name == FIG_TYPE_INEQ_DECOMP:
                ax.set_xlabel("Fraction of rejected users ($\\tau$)")
                ax.set_ylabel("Unfairness ($\mathcal{E}^2$)")
                #ax.set_yscale("log")
                h_dim_pdf = 14
            elif fig_type_name == FIG_TYPE_CONSTRAINT_INEQ_DECOMP:
                ax.invert_xaxis()
                ax.lines = ax.lines[::-1]
            #    ax.set_xlabel("Covariance threshold")
            #    ax.set_ylabel("Unfairness ($\mathcal{E}_2$)")
            #    #ax.set_ylim([0, .01])
            #    #ax.set_yscale("log")
            elif fig_type_name == FIG_TYPE_INTERGROUP_INEQ:
                #ax.set_xlabel("Fraction of rejected users")
                #ax.set_ylabel("Between-group unfairness ($\mathcal{E}_2$)")
                h_dim_pdf = 13
            elif fig_type_name == FIG_TYPE_INTERGROUP_SPLITS:
                h_dim_pdf = 14
                #for tick in ax.get_xticklabels():
                #    tick.set_rotation(90)

            fig.set_size_inches(h_dim_pdf, 7)
            fig.tight_layout()
            if output_channel == 'upload':
                # PDF
                file_path = out_dir + get_figure_name(dataset,
                        fig_type_name, fig_name)
                if len(ax.lines) > 1 and separate_legend:
                    axes = [ax]
                    if ax2 is not None:
                        axes.append(ax2)
                    for i, ax_obj in enumerate(axes):
                        fig_legend = plt.figure(figsize=(h_dim_pdf, 1.2))
                        handles, labels = ax_obj.get_legend_handles_labels()
                        fig_legend.legend(handles, labels, 'center', ncol=2)
                        fig_legend.savefig(file_path + "_legend{}.pdf".format(i))
                else:
                    ax.legend()
                fig.savefig(file_path + '.pdf')

                # PNG
                ax.set_title(fig_name)
                if len(ax.lines) > 1:
                    ax.legend()
                if ax2 is not None:
                    ax2.legend()
                fig.set_size_inches(12, 12)
                fig.tight_layout()
                fig.savefig(file_path + '.png')
            elif output_channel == 'show':
                #fig.show()
                ax.set_title(fig_name)
                plt.show()

    if output_channel == 'upload':
        output.upload_results([out_dir],
                'results/',
                'inequality_indices/measuring_inequality',
                file_extension_filter='.png')
    #plt.clf()
    plt.close('all')

def write_figure_links(wiki_file, out_dir, dataset, fig_type):
    web_target_dir = get_web_target_dir(out_dir)
    for fig_name in figures[fig_type]:
        full_fig_name = get_figure_name(dataset,
                fig_type, fig_name) + '.png'
        web_postfix = web_target_dir + full_fig_name
        wiki_file.write(output.web_attachment(web_postfix) + '\n\n')

def get_wiki_file(out_dir):
    wiki_file_loc = out_dir + 'res_wiki.txt'
    with open(wiki_file_loc, 'w') as wiki_file:
        wiki_file.write("== Results ==\n\n")
    return wiki_file_loc

def emit_lorenz_curves(wiki_file, out_dir, dataset, lorenz_desc):
    wiki_file.write("The Lorenz curve is computed over the {}\n\n".format(lorenz_desc))
    write_figure_links(wiki_file, out_dir, dataset, FIG_TYPE_LORENZ)

def emit_acc_fairness_curves(wiki_file, out_dir, dataset):
    wiki_file.write("Accuracy fairness tardeoff curves. Users are ranked based on their score, then the acceptance threshold is varied from 0 to n and for each threshold the accuracy and (un)fairness values of the resulting classifier are plotted.\n\n")
    write_figure_links(wiki_file, out_dir, dataset, FIG_TYPE_ACC_FAIR)

def emit_curves(wiki_file, out_dir, dataset, fig_type, description):
    wiki_file.write(description + "\n\n")
    write_figure_links(wiki_file, out_dir, dataset, fig_type)

def write_wiki_results(wiki_file, result_labels, results,
        results_format, hyperparams, method_infos):

    wiki_file.write("=== Accuracy and inequality numbers: ===\n\n")
    res_metrics = [[method] + metrics for method, metrics in \
            results.items()]
    output.write_table(wiki_file, ['Method'] + result_labels,
            res_metrics, val_format=[''] + results_format)

    # Index rankings
    wiki_file.write("\n'''Method rankings (best to worst performing):'''\n\n")
    for position, index in enumerate(result_labels):
        method_scores = [(method_res[0],
            method_res[position + 1]) for method_res in \
            res_metrics]
        method_scores = sorted(method_scores,
                key=lambda method_score: abs(method_score[1]))
        ranking = ", ".join(method_score[0] for method_score \
                in method_scores)
        wiki_file.write(" * {}: {}\n".format(index, ranking))

    wiki_file.write('\n== Methods ==\n\n')
    for method, params in hyperparams.items():
        #wiki_file.write('Hyperparameters are selected using 5-fold cross-validation on the training set.\n\n')
        method_info_link = output.get_wiki_link(
                *method_infos[method])
        wiki_file.write(' * {}: {}'.format(method, method_info_link))
        if params:
            wiki_file.write(' with parameters:\n')
            for param_name, param_val in params.items():
                wiki_file.write('  * {} = {}\n'.format(param_name,
                    param_val))
        else:
            wiki_file.write('\n')


def compute_sensitive_groups(sens_feature_list, x_control, num_users=None):
    if num_users is None:
        any_group = next(iter(x_control.values()))
        num_users = len(any_group)

    if sens_feature_list is None:
        sens_feature_list = list(x_control.keys())

    if not sens_feature_list:
        return {'All': np.ones(num_users, dtype=bool)}

    sens_entry_list = defaultdict(list)
    for sens_feature in sens_feature_list:
        for sens_entry in x_control:
            if sens_entry.startswith(sens_feature + '_'):
                sens_entry_list[sens_feature].append(sens_entry[len(sens_feature)+1:])
            elif sens_entry == sens_feature:
                sens_entry_list[''].append(sens_entry)

    per_feature_groups = {}
    for sens_feature, sens_entries in sens_entry_list.items():
        if len(sens_entries) == 1:
            sens_entry = sens_entries[0]
            inv_entries = {"Female": "Male", "Male": "Female"}
            inv_entry = inv_entries.get(sens_entry, "non-" + sens_entry)
            sens_key = '_'.join([sens_feature, sens_entry]) if sens_feature != '' else sens_entry
            sens_group = x_control[sens_key].astype(bool)
            sens_groups = {sens_entry: sens_group,
                    inv_entry: ~sens_group}
            assert all(np.logical_xor(sens_group, ~sens_group))
        else:
            sens_groups = {sens_entry: x_control['_'.join(
                [sens_feature, sens_entry])].astype(bool) \
                    for sens_entry in sens_entries}
        per_feature_groups[sens_feature] = sens_groups

    init_sens_feature, init_sens_groups = next(iter(per_feature_groups.items()))
    intersection_groups = init_sens_groups
    del per_feature_groups[init_sens_feature]

    for sens_feature, sens_groups in per_feature_groups.items():
        new_isgs = {}
        for isg_name, isg in intersection_groups.items():
            for sg_name, sg in sens_groups.items():
                new_isg_name = isg_name + ', ' + sg_name
                new_isg = isg & sg
                new_isgs[new_isg_name] = new_isg
        intersection_groups = new_isgs

    # Filter out empty intersection groups
    intersection_groups = {isg_name: isg for isg_name, isg in intersection_groups.items() if sum(isg) > 0}
    
    mask = np.zeros(num_users, dtype=bool)
    for isg_name, isg in intersection_groups.items():
        assert not any(mask[isg]), "Overlap between existing groups and {}".format(isg_name)
        mask |= isg
    assert all(mask)

    return intersection_groups

def pretty_features(features):
    pretties = {"sex": "gender", "race_Caucasian": "Whites"}

    if isinstance(features, str):
        return pretties.get(features, features)
    return [pretties.get(feature, feature) for feature in features]

def powerset(base_list, include_empty=False):
    s = list(base_list)
    ps = list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))
    if not include_empty:
        ps.remove(())
    return ps

def compute_intersectional_inequalities(split_features, ds_feature_groups, benefits):
        split_combinations = []
        group_amounts = []
        intergroup_inequality_fracs = []
        intragroup_inequality_fracs = []

        for feature_comb in powerset(split_features):
            #print("Splitting on features", feature_comb)
            split_combinations.append(feature_comb)

            feature_groups = compute_sensitive_groups(
                    feature_comb, ds_feature_groups)
            group_amounts.append(len(feature_groups))

            decomp = get_inequality_decomp(benefits, feature_groups)
            overall_ineq = decomp['overall_ineq']
            intergroup_ineq = decomp['intergroup_ineq']
            group_ineqs = {g: gie / overall_ineq * 100 for g, gie in zip(decomp['group_names'], decomp['subgroup_ineqs'])}
            assert intergroup_ineq <= overall_ineq
            intergroup_fraction = intergroup_ineq / overall_ineq
            intergroup_inequality_fracs.append(intergroup_fraction * 100)
            intragroup_inequality_fracs.append(group_ineqs)


        intersectional_labels = ["{}\n({})".format("\n".join(pretty_features(feature_comb)), group_amount) \
                for feature_comb, group_amount in \
                zip(split_combinations, group_amounts)]

        return intersectional_labels, intergroup_inequality_fracs, intragroup_inequality_fracs

def create_feature_groups(split_features, feature_names, x, x_control):
    name_values = {fn: fc for fn, fc in zip(feature_names, x.T)}
    for fn, fc in x_control.items():
        if fn not in name_values:
            name_values[fn] = fc

    ds_features_groups = {}
    # TODO: binarize continous features or handle otherwise
    for fn, fc in name_values.items():
        for feature in split_features:
            if fn.startswith(feature):
                ds_features_groups[fn] = fc

    return ds_features_groups

