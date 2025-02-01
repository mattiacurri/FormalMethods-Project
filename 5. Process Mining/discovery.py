import pm4py
import matplotlib.pyplot as plt
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.visualization.heuristics_net import visualizer as hn_visualizer
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
import pandas as pd
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.filtering.log.variants.variants_filter import get_variants
import pm4py.write as write_xes
from pm4py.algo.evaluation.replay_fitness import algorithm as fitness
from pm4py.algo.evaluation.precision import algorithm as precision
from pm4py.algo.evaluation.simplicity import algorithm as simplicity
from pm4py.algo.evaluation.generalization import algorithm as generalization
import pm4py.algo.analysis.workflow_net.algorithm as wf_net
from pm4py.algo.evaluation.replay_fitness.algorithm import Variants

def format_csv(input_file, output_file):
    """
    This function reads a CSV file and formats it in a specific way.
    Args:
        input_file (str): Name of the input CSV file
        output_file (str): Name of the output CSV file
    """
    formatted_rows = []
    try:
        with open(input_file, 'r') as file:
            lines = file.readlines()
            #Add the header
            formatted_rows.append("Timestamp, CaseID, Activity")
            # Format each line in the file
            for line in lines:
                line = line.strip()
                if not line:  # Ignore empty lines
                    continue
                elements = line.split()
                if len(elements) >= 4:  
                    timestamp = f"{elements[0]} {elements[1]}"
                    case_id = elements[2]
                    activity = ' '.join(elements[3:])
                    formatted_row = f"{timestamp}, {case_id}, {activity}"
                    formatted_rows.append(formatted_row)
        # Write the formatted rows to the output file
        with open(output_file, 'w', newline='') as file:
            for row in formatted_rows:
                file.write(row + '\n')
    except Exception as e:
        print(f"A problem occurred while formatting the CSV file: {e}")
        

def csv_to_xes(input_file, trimmed=False):
    """
    This function reads a CSV file and converts it to a XES file.
    Args:
        input_file (str): Name of the input CSV file
    """
    data = pd.read_csv(input_file, sep=",")
    cols = ['time:timestamp', 'case:concept:name','concept:name']
    data.columns = cols
    data['time:timestamp'] = pd.to_datetime(data['time:timestamp'])
    data['concept:name'] = data['concept:name'].astype(str)

    # take the subset of columns where time:timestamp, case:concept:name and concept:name are unique
    data = data.drop_duplicates(subset=['time:timestamp', 'case:concept:name', 'concept:name'])

    log = log_converter.apply(data, variant=log_converter.Variants.TO_EVENT_LOG)
    if trimmed:
        # Remove traces with less than 4 events
        l = get_variants(log)
        print(f'----- Number of variants: {len(l)}')
        l = [x for x in l if len(x) > 3]
        log = pm4py.algo.filtering.log.variants.variants_filter.apply(log, l)
        print(f'----- Number of variants after trimming: {len(l)}')
    
    output_name = input_file.split(".")[0] + ".xes"
    write_xes.write_xes(log, output_name)


import csv
from datetime import datetime, timedelta
from collections import Counter

def process_csv(input_file, output_file):
    with open(input_file, 'r') as f:
        reader = csv.reader(f, skipinitialspace=True)
        header = next(reader)
        activity_records = []
        
        current_activity = None
        current_case = None
        start_timestamp = None
        
        for row in reader:
            timestamp = datetime.strptime(row[0].strip(), '%Y-%m-%d %H:%M:%S')
            case_id = row[1].strip()
            activity = row[2].strip()
            
            if activity != current_activity or case_id != current_case:
                # Save previous activity window if it exists
                if current_activity is not None and start_timestamp is not None:
                    activity_records.append((start_timestamp, current_case, current_activity))
                    # Only append end timestamp if different from start
                    if last_timestamp != start_timestamp:
                        activity_records.append((last_timestamp, current_case, current_activity))
                
                # Start new activity window
                current_activity = activity
                current_case = case_id
                start_timestamp = timestamp
            
            # Update last timestamp for current window
            last_timestamp = timestamp
            
        # Handle the last activity window
        if current_activity is not None and start_timestamp is not None:
            activity_records.append((start_timestamp, current_case, current_activity))
            # Only append end timestamp if different from start
            if last_timestamp != start_timestamp:
                activity_records.append((last_timestamp, current_case, current_activity))

    # Write results sorted by timestamp
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        activity_records.sort() # Sort by timestamp
        for record in activity_records:
            writer.writerow(record)

def remove_suf_num(input_file='input.csv', output_file='output.csv'):
    """Elabora un file CSV rimuovendo i suffissi numerici da CaseID e Activity."""
    
    def generalize_label(label):
        """
        Generalizes activity labels by removing numerical suffixes or grouping similar activities.
        This function processes activity labels to create more general categories by:
        1. Removing numerical suffixes (e.g., 'activity_1' becomes 'activity')
        2. Grouping similar activities (e.g., 'sink_faucet_-_hot' and 'sink_faucet_-_cold')
        Parameters
        ----------
        label : str
            The activity label to be generalized.
        Returns
        -------
        str
            The generalized label. If the input is 'None' or doesn't match any 
            generalization pattern, returns the original label.
        Examples
        --------
        >>> generalize_label('activity_1')
        'activity'
        >>> generalize_label('sink_faucet_-_hot')
        'sink_faucet'
        >>> generalize_label('None')
        'None'
        """
        """Unisce le etichette rimuovendo il suffisso numerico finale."""
        if label == 'None':
            return label
        # group sink_faucet_-_hot and sink_faucet_-_cold
        # Group similar faucet activities
        if 'Sink_faucet_-_hot' in label or 'Sink_faucet_-_cold' in label:
            return 'Sink_faucet'
        
        parts = label.rsplit('_', 1)
        if len(parts) > 1 and parts[-1].isdigit():
            return parts[0]

        return label

    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile, skipinitialspace=True)
        writer = csv.writer(outfile)
        
        # Processa header
        writer.writerow(next(reader))
        
        # Processa righe
        for row in reader:
            # CaseID processing
            original_caseid = row[1].strip()
            new_caseid = generalize_label(original_caseid)
            
            # Activity processing
            activity_parts = row[2].strip().split()
            if len(activity_parts) >= 3:
                original_device, action = activity_parts[0], activity_parts[1]
                new_device = generalize_label(original_device)
                new_activity = f"{new_device} {action} {' '.join(activity_parts[2:])}"
            else:
                new_activity = row[2].strip()
            
            # Scrittura riga modificata
            writer.writerow([row[0].strip(), new_caseid, new_activity])


def plot_metrics(alpha_metrics, heuristic_metrics, inductive_metrics, img_name='metric/metrics.png'):
    """
    This function plots the metrics of different process discovery algorithms.
    Args:
        alpha_metrics (tuple): Tuple containing the metrics of the Alpha Miner
        heuristic_metrics (tuple): Tuple containing the metrics of the Heuristic Miner
        inductive_metrics (tuple): Tuple containing the metrics of the Inductive Miner
    """
    algorithms = ['Alpha Miner', 'Heuristic Miner', 'Inductive Miner']
    metrics = ['Fitness', 'Precision', 'Simplicity', 'Generalization']
    
    # Replace None with 0 for plotting
    def handle_none(metric):
        return [0 if x is None else x for x in metric]

    values = [
        handle_none([alpha_metrics[0], heuristic_metrics[0], inductive_metrics[0]]),  # Fitness
        handle_none([alpha_metrics[1], heuristic_metrics[1], inductive_metrics[1]]),  # Precision
        handle_none([alpha_metrics[2], heuristic_metrics[2], inductive_metrics[2]]),  # Simplicity
        handle_none([alpha_metrics[3], heuristic_metrics[3], inductive_metrics[3]])   # Generalization
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for i, metric in enumerate(metrics):
        bars = axes[i].barh(algorithms, values[i], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[i].set_title(metric)
        axes[i].set_xlabel("Value")
        axes[i].set_xlim(0, 1)
        # Add value labels to the right of each bar
        for bar in bars:
            width = bar.get_width()
            axes[i].text(width, bar.get_y() + bar.get_height()/2.,
                     f'{width:.3f}',
                     ha='left', va='center')
        
    plt.tight_layout()
    plt.savefig(img_name)
    plt.close()

#  variant=Variants.TOKEN_BASED
def heuristic_mining(train, test, img_name='images/heuristic_miner.png', **params):
    """
    This function performs the Heuristic Miner algorithm on an event log.
    Args:
        log (str): Path to the XES file
        Returns:
        tuple: Tuple containing the metrics of the Heuristic Miner
    """
    print("----- Heuristic Miner Process")
    
    net, initial_marking, final_marking = heuristics_miner.apply(train) # , parameters={"heu_net_decoration": "frequency"})
    gviz = pn_visualizer.apply(net, initial_marking, final_marking)
    pn_visualizer.save(gviz, img_name)
    
    print("----- Heuristic Miner Evaluation")
    fitness_value = fitness.apply(test, net, initial_marking, final_marking, variant=params['variant'])["average_trace_fitness"]  # Direct dictionary
    precision_value = precision.apply(test, net, initial_marking, final_marking)  # Direct float value
    simplicity_value = simplicity.apply(net)
    generalization_value = generalization.apply(test, net, initial_marking, final_marking)
    
    l=[fitness_value, precision_value, simplicity_value, generalization_value]
    return l


def inductive_mining(train, test, img_name='images/inductive_miner.png', **params):
    """
    This function performs the Inductive Miner algorithm on an event log.
    Args:
        log (str): Path to the XES file
    Returns:
        tuple: Tuple containing the metrics of the Inductive Miner
    """
    print("----- Inductive Miner Process")
    
    net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(train, noise_threshold=params['noise_threshold']) ## , noise_threshold=0.5)
    gviz = pn_visualizer.apply(net, initial_marking, final_marking)
    pn_visualizer.save(gviz, img_name)

    print("----- Inductive Miner Evaluation")
    
    fitness_value = fitness.apply(test, net, initial_marking, final_marking, variant=params['variant'])["average_trace_fitness"]  # Direct dictionary
    precision_value = precision.apply(test, net, initial_marking, final_marking)  # Direct float value
    simplicity_value = simplicity.apply(net)
    generalization_value = generalization.apply(test, net, initial_marking, final_marking)
    
    l=[fitness_value, precision_value, simplicity_value, generalization_value]
    return l

# , **params
def alpha_mining(train, test, img_name='images/alpha_miner.png', **params):
    """
    This function performs the Alpha Miner algorithm on an event log and evaluates the model.
    Args:
        log (str): Path to the XES file
    Returns:
        tuple: Tuple containing the metrics of the Alpha Miner
    """
    print("----- Alpha Miner Process")
    # Apply Alpha Miner to discover the Petri net
    net, initial_marking, final_marking = alpha_miner.apply(train)
    gviz = pn_visualizer.apply(net, initial_marking, final_marking)
    pn_visualizer.save(gviz, img_name)
    
    print("----- Alpha Miner Evaluation")
    fitness_value = fitness.apply(test, net, initial_marking, final_marking, variant=params['variant'])["average_trace_fitness"]  # Direct dictionary
    precision_value = precision.apply(test, net, initial_marking, final_marking)  # Direct float value
    simplicity_value = simplicity.apply(net)
    generalization_value = generalization.apply(test, net, initial_marking, final_marking)
    
    l=[fitness_value, precision_value, simplicity_value, generalization_value]
    return l

# BASE EXPERIMENT - NO PROCESSING, BASE MODEL
format_csv("mit/mit.csv", "mit/mit_formatted_noproc_notrimmed.csv")
#remove_suf_num('mit/mit_formatted.csv', 'mit/mit_formatted_ts.csv')
#process_csv('mit/mit_formatted_ts.csv', 'mit/mit_formatted_ts_cleaned.csv')
csv_to_xes("mit/mit_formatted_noproc_notrimmed.csv", False)

input_file = "mit/mit_formatted_noproc_notrimmed.xes"

log = pm4py.read_xes(input_file)

# Execute the mining
params = {'variant': Variants.TOKEN_BASED, 'noise_threshold': 0.0}
alpha_metrics = alpha_mining(log, 'images/alpha_miner_noproc_base_notrimmed.png', **params)   
heuristic_metrics = heuristic_mining(log, 'images/heuristic_miner_noproc_base_notrimmed.png', **params)
inductive_metrics = inductive_mining(log, 'images/inductive_miner_noproc_base_notrimmed.png', **params) 

plot_metrics(alpha_metrics, heuristic_metrics, inductive_metrics, 'metric/metrics_noproc_base_notrimmed.png')



# BASE EXPERIMENT - NO PROCESSING, TUNED MODEL
format_csv("mit/mit.csv", "mit/mit_formatted_noproc_tuned_notrimmed.csv")
#remove_suf_num('mit/mit_formatted.csv', 'mit/mit_formatted_ts.csv')
#process_csv('mit/mit_formatted_ts.csv', 'mit/mit_formatted_ts_cleaned.csv')
csv_to_xes("mit/mit_formatted_noproc_tuned_notrimmed.csv", False)

input_file = "mit/mit_formatted_noproc_tuned_notrimmed.xes"

log = pm4py.read_xes(input_file)

# Execute the mining
params = {'variant': Variants.TOKEN_BASED, 'noise_threshold': 0.0}
alpha_metrics = alpha_mining(log, 'images/alpha_miner_noproc_tuned_notrimmed.png', **params)   
heuristic_metrics = heuristic_mining(log, 'images/heuristic_miner_noproc_tuned_notrimmed.png', **params)

for n in [0.1, 0.2, 0.3, 0.5, 0.6, 0.8]:
    params = {'variant': Variants.TOKEN_BASED, 'noise_threshold': n}
    inductive_metrics = inductive_mining(log, f'images/inductive_miner_noproc_tuned_{n}_notrimmed.png', **params)

    plot_metrics(alpha_metrics, heuristic_metrics, inductive_metrics, f'metric/metrics_noproc_tuned_{n}_notrimmed.png')



# PROCESSING, BASE MODEL, NO TRIM
format_csv("mit/mit.csv", "mit/mit_formatted_proc_notrimmed.csv")
remove_suf_num('mit/mit_formatted_proc_notrimmed.csv', 'mit/mit_formatted_proc_ts_notrimmed.csv')
process_csv('mit/mit_formatted_proc_ts_notrimmed.csv', 'mit/mit_formatted_proc_ts_cleaned_notrimmed.csv')
csv_to_xes("mit/mit_formatted_proc_ts_cleaned_notrimmed.csv", False)

input_file = "mit/mit_formatted_proc_ts_cleaned_notrimmed.xes"

log = pm4py.read_xes(input_file)

# Execute the mining
params = {'variant': Variants.TOKEN_BASED, 'noise_threshold': 0.0}
alpha_metrics = alpha_mining(log, 'images/alpha_miner_proc_base_notrimmed.png', **params)   
heuristic_metrics = heuristic_mining(log, 'images/heuristic_miner_proc_base_notrimmed.png', **params)
inductive_metrics = inductive_mining(log, 'images/inductive_miner_proc_base_notrimmed.png', **params) 

plot_metrics(alpha_metrics, heuristic_metrics, inductive_metrics, 'metric/metrics_proc_base_notrimmed.png')

# PROCESSING, TUNED MODEL, NO TRIM
format_csv("mit/mit.csv", "mit/mit_formatted_proc_notrimmed.csv")
remove_suf_num('mit/mit_formatted_proc_notrimmed.csv', 'mit/mit_formatted_proc_ts_notrimmed.csv')
process_csv('mit/mit_formatted_proc_ts_notrimmed.csv', 'mit/mit_formatted_proc_ts_cleaned_notrimmed.csv')
csv_to_xes("mit/mit_formatted_proc_ts_cleaned_notrimmed.csv", False)

input_file = "mit/mit_formatted_proc_ts_cleaned_notrimmed.xes"

log = pm4py.read_xes(input_file)

# Execute the mining
params = {'variant': Variants.TOKEN_BASED, 'noise_threshold': 0.0}
alpha_metrics = alpha_mining(log, f'images/alpha_miner_proc_tuned_notrimmed.png', **params)   
heuristic_metrics = heuristic_mining(log, f'images/heuristic_miner_proc_tuned_notrimmed.png', **params)
for n in [0.1, 0.2, 0.3, 0.5, 0.6, 0.8]:
    params = {'variant': Variants.TOKEN_BASED, 'noise_threshold': n}
    inductive_metrics = inductive_mining(log, f'images/inductive_miner_proc_tuned_{n}_notrimmed.png', **params)

    plot_metrics(alpha_metrics, heuristic_metrics, inductive_metrics, f'metric/metrics_proc_tuned_{n}_notrimmed.png')



# NO PROCESSING, TUNED MODEL, TRIM
format_csv("mit/mit.csv", "mit/mit_formatted_noproc_tuned_trimmed.csv")
#remove_suf_num('mit/mit_formatted.csv', 'mit/mit_formatted_ts.csv')
#process_csv('mit/mit_formatted_ts.csv', 'mit/mit_formatted_ts_cleaned.csv')
csv_to_xes("mit/mit_formatted_noproc_tuned_trimmed.csv", True)

input_file = "mit/mit_formatted_noproc_tuned_trimmed.xes"

log = pm4py.read_xes(input_file)

# Execute the mining
params = {'variant': Variants.TOKEN_BASED, 'noise_threshold': 0.0}
alpha_metrics = alpha_mining(log, 'images/alpha_miner_noproc_tuned_trimmed.png', **params)   
heuristic_metrics = heuristic_mining(log, 'images/heuristic_miner_noproc_tuned_trimmed.png', **params)

for n in [0.1, 0.2, 0.3, 0.5, 0.6, 0.8]:
    params = {'variant': Variants.TOKEN_BASED, 'noise_threshold': n}
    inductive_metrics = inductive_mining(log, f'images/inductive_miner_noproc_tuned_{n}_trimmed.png', **params)

    plot_metrics(alpha_metrics, heuristic_metrics, inductive_metrics, f'metric/metrics_noproc_tuned_{n}_trimmed.png')



# PROCESSING, TUNED MODEL, TRIM
format_csv("mit/mit.csv", "mit/mit_formatted_proc_trimmed.csv")
remove_suf_num('mit/mit_formatted_proc_trimmed.csv', 'mit/mit_formatted_proc_ts_trimmed.csv')
process_csv('mit/mit_formatted_proc_ts_trimmed.csv', 'mit/mit_formatted_proc_ts_cleaned_trimmed.csv')
csv_to_xes("mit/mit_formatted_proc_ts_cleaned_trimmed.csv", True)

input_file = "mit/mit_formatted_proc_ts_cleaned_trimmed.xes"

log = pm4py.read_xes(input_file)

# Execute the mining
params = {'variant': Variants.TOKEN_BASED, 'noise_threshold': 0.0}
alpha_metrics = alpha_mining(log, f'images/alpha_miner_proc_tuned_trimmed.png', **params)   
heuristic_metrics = heuristic_mining(log, f'images/heuristic_miner_proc_tuned_trimmed.png', **params)
for n in [0.1, 0.2, 0.3, 0.5, 0.6, 0.8]:
    params = {'variant': Variants.TOKEN_BASED, 'noise_threshold': n}
    inductive_metrics = inductive_mining(log, f'images/inductive_miner_proc_tuned_{n}_trimmed.png', **params)

    plot_metrics(alpha_metrics, heuristic_metrics, inductive_metrics, f'metric/metrics_proc_tuned_{n}_trimmed.png')



# NO PROCESSING, BASE MODEL, TRIM
format_csv("mit/mit.csv", "mit/mit_formatted_noproc_trimmed.csv")
#remove_suf_num('mit/mit_formatted.csv', 'mit/mit_formatted_ts.csv')
#process_csv('mit/mit_formatted_ts.csv', 'mit/mit_formatted_ts_cleaned.csv')
csv_to_xes("mit/mit_formatted_noproc_trimmed.csv", True)

input_file = "mit/mit_formatted_noproc_trimmed.xes"

log = pm4py.read_xes(input_file)

# Execute the mining
params = {'variant': Variants.TOKEN_BASED, 'noise_threshold': 0.0}
alpha_metrics = alpha_mining(log, 'images/alpha_miner_noproc_base_trimmed.png', **params)   
heuristic_metrics = heuristic_mining(log, 'images/heuristic_miner_noproc_base_trimmed.png', **params)
inductive_metrics = inductive_mining(log, 'images/inductive_miner_noproc_base_trimmed.png', **params) 

plot_metrics(alpha_metrics, heuristic_metrics, inductive_metrics, 'metric/metrics_noproc_base_trimmed.png')

# PROCESSING, BASE MODEL, TRIM
format_csv("mit/mit.csv", "mit/mit_formatted_proc_trimmed.csv")
remove_suf_num('mit/mit_formatted_proc_trimmed.csv', 'mit/mit_formatted_proc_ts_trimmed.csv')
process_csv('mit/mit_formatted_proc_ts_trimmed.csv', 'mit/mit_formatted_proc_ts_cleaned_trimmed.csv')
csv_to_xes("mit/mit_formatted_proc_ts_cleaned_trimmed.csv", True)

input_file = "mit/mit_formatted_proc_ts_cleaned_trimmed.xes"

log = pm4py.read_xes(input_file)

# Execute the mining
params = {'variant': Variants.TOKEN_BASED, 'noise_threshold': 0.0}
alpha_metrics = alpha_mining(log, 'images/alpha_miner_proc_base_trimmed.png', **params)   
heuristic_metrics = heuristic_mining(log, 'images/heuristic_miner_proc_base_trimmed.png', **params)
inductive_metrics = inductive_mining(log, 'images/inductive_miner_proc_base_trimmed.png', **params) 

plot_metrics(alpha_metrics, heuristic_metrics, inductive_metrics, 'metric/metrics_proc_base_trimmed.png')


# LLMs experiments
input_file = "mit/mit_formatted_proc_ts_cleaned_trimmed.xes"

log = pm4py.read_xes(input_file)

# frequency of activities in csv
# load the csv file
data = pd.read_csv('mit/mit_formatted_noproc_notrimmed.csv')
# get the activity column
print(data.columns)
activity_column = data[" Activity"]

# count the frequency of each activity
activity_frequency = Counter(activity_column)
print(activity_frequency)

print(len(activity_frequency))

# take all the rows that have activity that appear only once
unique_activity = data[data[" Activity"].map(activity_frequency) == 1]

print(unique_activity)

# save the unique activity to a csv file
unique_activity.to_csv('mit/unique_activity.csv', index=False)


data = pd.read_csv('mit/mit_formatted_noproc_notrimmed.csv')
for llm in ['claude', 'gpt', 'mistral', 'qwen', 'deepseek']:
    llm_data = pd.read_csv(f'mit/aug/{llm}.csv')

    input_file = pd.concat([data, llm_data])

    # save to csv
    input_file.to_csv(f'mit/mit_{llm}.csv', index=False)
    
    # convert to xes
    csv_to_xes(f'mit/mit_{llm}.csv', False)
    log = pm4py.read_xes(f'mit/mit_{llm}.xes')
    log_train = pm4py.read_xes('mit/mit_formatted_noproc_notrimmed.xes')

    params = {'variant': Variants.TOKEN_BASED, 'noise_threshold': 0.0}

    alpha_metrics = alpha_mining(log,log_train, f'images/alpha_miner_{llm}.png', **params)
    heuristic_metrics = heuristic_mining(log,log_train, f'images/heuristic_miner_{llm}.png', **params)

    for n in [0.0, 0.2, 0.5, 0.6, 0.8]:
        params = {'variant': Variants.TOKEN_BASED, 'noise_threshold': n}
        inductive_metrics = inductive_mining(log,log, f'images/inductive_miner_{llm}_{n}.png', **params)
        plot_metrics(alpha_metrics, heuristic_metrics, inductive_metrics, f'metric/{llm}_{n}_augdataset.png')

