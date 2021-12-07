import pandas as pd
import numpy as np

def kmeans(X, k):  
    diff = 1  
    cluster = np.zeros(X.shape[0])  
    centroids = X.sample(n=k).values  
    while diff:     
        # for each observation     
        for i, row in enumerate(X):         
            mn_dist = float('inf')        
            # dist of the point from all centroids        
            for idx, centroid in enumerate(centroids):            
                d = np.sqrt((centroid[0]-row[0])**2 + (centroid[1]-row[1])**2)
                # store closest centroid            
                if mn_dist > d:               
                    mn_dist = d               
                    cluster[i] = idx     
                    new_centroids = pd.DataFrame(X).groupby(by=cluster).mean().values     
                    # if centroids are same then leave     
                    if np.count_nonzero(centroids-new_centroids) == 0:        
                        diff = 0      
                    else:        
                        centroids = new_centroids  
    return centroids, cluster