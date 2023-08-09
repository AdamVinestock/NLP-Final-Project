import pandas as pd

def calc_mean_ppx_instance(human_responses, machine_responses):
    """
    Each instance in the dataset has a unique "name" value with a different number of sentences for human and machine responses
    This method calculates the mean perplexity for each instance
    :param human_responses: df with responses
    :param machine_responses: df with responses
    :return: human and machine dataframe containing the mean perplexity for each instance
    """

    h_grouped_mean = human_responses.groupby('name')['response'].mean().reset_index()
    h_sorted_df = h_grouped_mean.sort_values(by='name', ascending=True)

    m_grouped_mean = machine_responses.groupby('name')['response'].mean().reset_index()
    m_sorted_df = m_grouped_mean.sort_values(by='name', ascending=True)

    return h_sorted_df, m_sorted_df

def calc_diff_ppx_instance(human_responses, machine_responses):
    """
    input: human and machine mean responses sorted my 'name' (samples)
    returns df holding name and the difference between human and machine responses
    Note: the larger the difference the better the context policy
    """
    diff_df = human_responses.copy()
    diff_df["response"] = human_responses["response"] - machine_responses["response"]
    return diff_df

def calc_h_m_diff(human_path, machine_path):
    """
    input: paths of human and machine df holding responses for each sentence
    output: human-machine mean perplexity difference
    """
    h_df = pd.read_csv(human_path)     # human responses df with perplexity over all sentences
    m_df = pd.read_csv(machine_path)   # machine responses df with perplexity over all sentences
    h_sorted_df, m_sorted_df = calc_mean_ppx_instance(h_df, m_df) # mean perplexity for each instance
    diff_df = calc_diff_ppx_instance(h_sorted_df, m_sorted_df)    # human - machine perplexity for each instance
    return diff_df['response'].mean()



