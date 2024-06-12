def create_calendar_feature(data, column):
    '''
    Input:
    data: dataframe for processing
    column: single column. Interested datatime column

    Output:
    Create separate time feature columns for the dataset
    '''
    data['Time'] = data[column].astype(str).apply(lambda x: int(x.split()[-1].split(':')[0]))
    data['Year'] = data[column].astype(str).apply(lambda x: int(x.split()[0].split('-')[0]))
    data['Month'] = data[column].astype(str).apply(lambda x: int(x.split()[0].split('-')[1]))
    data['Day'] = data[column].astype(str).apply(lambda x: int(x.split()[0].split('-')[2]))

    pass


def target_correlation_rank(data, target, columns):
    '''
    Input:
    data: dataframe for processing
    target: target variable
    column: list of columns for correlation ranking

    Output:
    dict: A ranked dictionary of the correlation score
    '''
    
    # Initiate the dictionary
    corr_dict = dict()

    # Loop through the columns to get the value
    for col in columns:
        correlation_value = data[[target,col]].corr().values[0][1]
        corr_dict[col] = abs(correlation_value)
    
    # Return the dictionary in the reverse order
    return dict(sorted(corr_dict.items(), key=lambda item: item[1],reverse=True))
    

def variable_to_use(corr_dict, threshold = 0.3):
    '''
    Input:
    corr_dict: correlation value dictionary
    threshold: threshold value for filtering the variable

    Output:
    use_dict: a dictionary of variable above the threshold
    combine_dict: a dictionary of variable under the threshold
    '''

    use_dict = {k:v for k,v in corr_dict.items() if v>=threshold}
    combine_dict = {k:v for k,v in corr_dict.items() if v<threshold}

    return use_dict, combine_dict

def create_PCA(data, pca_component, variables_combine,threshold):
    '''
    Input:
    data: data to fit on
    pca_component: starting number of pca component to test on
    variables_combine: variables to make PCA on
    threshold: threshold value for filtering the variable

    Output:
    pca: PCA to transform the raw data
    pca_component: number of component of the PCA vector
    '''
    # import PCA
    from sklearn.decomposition import PCA
    # Initiate the variables
    explained = 0
    component_cnt = pca_component
    # Get the explained ratio
    if explained == 0:
        # Create PCA vector
        pca = PCA(n_components=component_cnt)
        pca.fit(data[variables_combine])
        # Sum up the explained ratio
        explained = sum(pca.explained_variance_ratio_)
    
    while explained<threshold:
        # component add 1
        component_cnt += 1
        # get PCA
        pca = PCA(n_components=component_cnt)
        pca.fit(data[variables_combine])
        # Sum up the explained ratio
        explained = sum(pca.explained_variance_ratio_)
    
    return pca, component_cnt
    

def create_cluster_list(data, use_variables):
    '''
    Input:
    data: data to fit on
    use variable: variables to create correlation & cluster on

    Output:
    variable list: list of list of variables in cluster
    '''
    # Import packages
    import scipy.cluster.hierarchy as spc
    # Correlation Values
    corr = data[use_variables].corr().values
    # Create the cluster number for the variable
    pdist = spc.distance.pdist(corr)
    linkage = spc.linkage(pdist, method='complete')
    idx = spc.fcluster(linkage, 0.5 * pdist.max(), 'distance')

    # Number of cluster
    num_cluster = len(set(idx))
    # Initiate the list to hold the variables
    variable_list = [[]] * num_cluster

    # loop through the idx
    for i in range(len(idx)):
        variable_list[idx[i]-1].append(use_variables[i])
    
    return variable_list