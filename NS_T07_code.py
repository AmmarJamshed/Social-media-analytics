#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------ #
#%% Topic 07: Network Links - Directions and Weights
# ------------------------------------------------------------ #
#
# ------------------------------------------------------------ #
# Set up environment
# ------------------------------------------------------------ #
#
import os
# https://networkx.org/documentation/stable/tutorial.html
import networkx as nx
# https://numpy.org/doc/stable/user/absolute_beginners.html#
from pyvis.network import Network
# https://pandas.pydata.org/docs/getting_started/intro_tutorials/index.html
import pandas as pd
# https://numpy.org/doc/stable/user/absolute_beginners.html#
import numpy as np
np.set_printoptions(suppress=True) # to avoid printing numbers with scientific notation
#
import math # to use sqrt and other mathematical functions
import matplotlib.pyplot as plt # to plot
from collections import Counter # to count number of instances
#
from bs4 import BeautifulSoup #  to parse data out of HTML and XML files
import urllib.request
#
nx.__version__
pd.__version__
np.__version__
#
str_cwd = '/Your_Path_To_Code'
os.chdir(str_cwd)
str_path = os.getcwd()
#
# ------------------------------------------------------------ #
# Define functions to be used
# ------------------------------------------------------------ #
#
# Function to plot a network using pyvis
def plot_G_pyvis(G, file_path = str_path + '/G.html', bln_weighted = False, bln_display_weights = False, height_px = '1000px', width_px = '1000px'):
    bln_directed = nx.is_directed(G)
    #
    nt = Network('1000px', '1000px', directed = bln_directed)
    nt.from_nx(G) # imports graph from networkx
    # nt.nodes is a list of dictionaries
    for n in nt.nodes:
        n['label'] = str(n['label'])
    #
    # nt.edges is a list of dictionaries
    if bln_weighted:
        for e in nt.edges:
            e['value'] = e['weight']
        if bln_display_weights:
            for e in nt.edges:
                e['label'] = str(e['value'])
    if bln_directed and bln_weighted:
        for e in nt.edges:
            e['arrowStrikethrough'] = False
    #
    nt.toggle_physics(False) # to avoid initial slow rendering
    # to activate forceAtlas2Based and interaction options
    nt.show_buttons(filter_=['physics', 'interaction'])
    nt.show(file_path)
#
# Extract links and titles from 'See Also' section of a Wiki page
def extract_wiki_links_and_titles(wiki_url):
    # Replace potentially problematic unicode character codes
    wiki_url = wiki_url.replace(u"\u2013", "-")
    wiki_url = wiki_url.replace("%E2%80%93", "-")
    #
    url = urllib.request.urlopen(wiki_url)
    soup = BeautifulSoup(url)
    sa = soup.find('span', {'id':'See_also'})
    if not sa:
        return []
    element = sa
    links = []
    while element.name != 'h2':
        if (element.name == 'a'):
            if element['href'].startswith('/wiki'):
                if not element['href'].startswith('/wiki/File:'):
                    str_title = element['title']
                    str_url = element['href']
                    # Replace potentially problematic unicode character codes
                    str_title = str_title.replace(u"\u2013", "-")
                    str_title = str_title.replace("%E2%80%93", "-")
                    str_url = str_url.replace(u"\u2013", "-")
                    str_url = str_url.replace("%E2%80%93", "-")
                    #
                    d_link = {
                        'title': str_title,
                        'url': 'https://en.wikipedia.org'+ str_url
                    }
                    links.append(d_link)
        element = element.next_element
    return links
#
# Build Wiki-page ego network of a given order
# extracts higher order nodes and links up to a given order
def G_wiki_ego(seed, seed_label, which_order = 1):
    G = nx.DiGraph()
    G.add_node(seed_label, label=seed_label, url=seed)
    #
    order = 1
    neighbors = extract_wiki_links_and_titles(seed)
    for n in neighbors:
        G.add_node(n['title'], label=n['title'], url=n['url'])
        G.add_edge(seed_label,n['title'])
    #
    while order < which_order:
        l_sn = []
        for n in neighbors:
            print(n['title'])
            second_neighbors = extract_wiki_links_and_titles(n['url'])
            for sn in second_neighbors:
                if not G.has_node(sn['title']):
                    l_sn.append(sn)
                    G.add_node(sn['title'], label=sn['title'], url=sn['url'])
                G.add_edge(n['title'], sn['title'])
        neighbors = l_sn
        order += 1
    #
    return G
#
# Extract categories from a Wiki page
def extract_wiki_categories(wiki_url):
    soup = BeautifulSoup(urllib.request.urlopen(wiki_url))
    cat = soup.find('a', {'title':'Help:Category'})
    if not cat or (cat.next_sibling is None):
        return []
    catlist = cat.next_sibling.next_sibling
    if catlist is None:
        return []
    cats = catlist.find_all('a')
    return [c['title'].split(':')[1] for c in cats]
#
# Calculate cosine similarity between two sets of tags
def cosine_sim(s1, s2):
    # & is a logical operator to compare sets (intersection between sets)
    return len(s1 & s2)/(math.sqrt(len(s1))*math.sqrt(len(s2)))
#
# Compute centrality concepts for *unrected*, *unweighted* graph and store in a pandas dataframe
def G_centrality_pd(G):
    C_degree = dict(nx.degree(G))
    C_close = dict(nx.closeness_centrality(G, wf_improved=True))
    C_btw = dict(nx.betweenness_centrality(G, normalized=True))
    #
    pd_C_degree = pd.DataFrame.from_dict(C_degree, orient='index', columns=['Degree']).sort_index()
    pd_C_close = pd.DataFrame.from_dict(C_close, orient='index', columns=['Closeness']).sort_index()
    pd_C_btw = pd.DataFrame.from_dict(C_btw, orient='index', columns=['Betweenness']).sort_index()
    #
    pd_C = pd.concat([pd_C_degree, pd_C_close, pd_C_btw], axis=1)
    return pd_C
#
# Compute centrality concepts for *directed*, *weighted* graph and store in a pandas dataframe
def D_centrality_pd(D):
    C_in_degree = dict(D.in_degree(weight='weight'))
    C_out_degree = dict(D.out_degree(weight='weight'))
    C_close = dict(nx.closeness_centrality(D.reverse(), distance = 'weight', wf_improved=True))
    C_PR = nx.pagerank(D, weight='weight')
    #
    pd_C_in_degree = pd.DataFrame.from_dict(C_in_degree, orient='index', columns=['InDegree']).sort_index()
    pd_C_out_degree = pd.DataFrame.from_dict(C_out_degree, orient='index', columns=['OutDegree']).sort_index()
    pd_C_close = pd.DataFrame.from_dict(C_close, orient='index', columns=['Closeness']).sort_index()
    pd_C_pr = pd.DataFrame.from_dict(C_PR, orient='index', columns=['PageRank']).sort_index()
    #
    pd_C = pd.concat([pd_C_in_degree, pd_C_out_degree, pd_C_close, pd_C_pr], axis=1)
    return pd_C
#
# ------------------------------------------------------------ #
# Directed ego-network using Wikipedia "See also" outgoing links
# ------------------------------------------------------------ #
seed = 'https://en.wikipedia.org/wiki/Social_media'
seed_label = 'Social media'
#
D = G_wiki_ego(seed, seed_label)
D.nodes(data=True)
D.number_of_nodes() # 29
D.number_of_edges() # 28
plot_G_pyvis(D, file_path=str_path + '/D_wiki_ego_o1.html')
#
D_o2 = G_wiki_ego(seed, seed_label, which_order = 2)
D_o2.nodes(data=True)
D_o2.number_of_nodes() # 242
D_o2.number_of_edges() # 260
plot_G_pyvis(D_o2, file_path=str_path + '/D_wiki_ego_o2.html')
#
D_o3 = G_wiki_ego(seed, seed_label, which_order = 3)
D_o3.nodes(data=True)
D_o3.number_of_nodes() # 1781
D_o3.number_of_edges() # 2278
plot_G_pyvis(D_o3, file_path=str_path + '/D_wiki_ego_o3.html')
#
# Build a k=2 core to ease visualisation
D = D_o3.copy()
D.remove_edges_from(nx.selfloop_edges(D))
D_k_core = nx.k_core(D, 2)
# len(D_k_core)
# len(D_k_core.edges)
plot_G_pyvis(D_k_core, str_path + '/D_wiki_ego_o3_k_2_core.html')
#
D_o3.has_node('Network science') # check if a term is in ego network
# check if there are paths between concept and ego
nx.has_path(D_o3, 'Social media', 'Network science') # This should be True
nx.has_path(D_o3, 'Network science', 'Social media') # True only if link is reciprocal
# What is the shortest path between ego and a concept?
nx.shortest_path(D_o3, 'Social media', 'Network science')
# build a sorted list of concepts in the ego network
l_concepts = []
for n in D_o3.nodes():
    l_concepts.append(D_o3.nodes[n]['label'])
l_concepts = sorted(l_concepts)
# find all terms containing a given word
str_word = 'software'
l_matches = list(filter(lambda f: str_word in f, l_concepts))
# find shortest path between seed and one of the terms found
nx.shortest_path(D_o3, 'Social media', 'Social network analysis software')
#
# ------------------------------------------------------------ #
# Use directed ego network to build a list of categories
# associated to each node (useful to assess topical locality)
# ------------------------------------------------------------ #
#
D = D_o2.copy()
# D = D_o3.copy()
N = D.number_of_nodes()
# extract categories from pages in ego network (may take a while)
categories = {} # dictionary of sets
i_page = 0
for page in D.nodes():
    i_page += 1
    print(page + ' (' + str(i_page) + '/' + str(N) + ')')
    categories[page] = set(extract_wiki_categories(D.nodes[page]['url']))
#
categories
#
# calculate pairwise similarities, avoiding duplication
sim = {} # dictionary with term-tuple as key and similarity score as value
for page_1 in D.nodes():
    for page_2 in D.nodes():
        if page_1 <= page_2:
            continue
        if len(categories[page_1]) == 0 or len(categories[page_2]) == 0:
            continue
        sim[(page_1,page_2)] = cosine_sim(categories[page_1], categories[page_2])
#
len(sim)
#
# sorted sequence of cosine similarity values
sim_sequence = sorted(round(v, 3) for v in sim.values())
# compute histogram with counts of cosine similarity values
counts, bins, patches = plt.hist(sim_sequence, bins=np.linspace(0, 1, 11, endpoint=True).round(1), density = False)
# translate result into a dictionary for readability
d_sim_hist = {}
for i in range(0,len(counts)):
    d_sim_hist[tuple(bins[i:(i+2)])] = counts[i]
#
d_sim_hist
# identify term-tuples with high cosine similarity
high_similarity = [key for key in sim if sim[key]>0.6]
categories[high_similarity[2][0]]
categories[high_similarity[2][1]]
#
# ------------------------------------------------------------ #
# Use directed ego network to see an application of PageRank
# ------------------------------------------------------------ #
#
# Compute centrality indicators for directed networks
pd_C = D_centrality_pd(D_o3)
top_n = 15
pd_C.nlargest(top_n,'InDegree', keep='all') # top_n largest entries by k_in
pd_C.nlargest(top_n,'PageRank', keep='all') # top_n largest entries by PageRank
pd_C.plot.scatter(x = 'InDegree', y='PageRank', logx = True, logy = True) # positive correlation, but there are many cases of the same k_in with varying PageRank
# PageRank is not the same as in-degree, as it also considers the sources of the incoming links
#
pd_C.nlargest(top_n,'OutDegree', keep='all') # top_n largest entries by k_out
pd_C.nlargest(top_n,'Closeness', keep='all') # top_n largest entries by closeness
pd_C.plot.scatter(x = 'OutDegree', y='Closeness', logx = True, logy = True) # positive correlation, but there are many cases of the same k_out with varying closeness
#
# ------------------------------------------------------------ #
# Twitter Networks
# ------------------------------------------------------------ #
#
# Creating a Twitter App requires access to Twitter's developer platform, which requires an application process with Twitter
# https://developer.twitter.com/en/portal/dashboard
# https://developer.twitter.com/en/portal/products/elevated
#
import json
from twython import Twython
# https://twython.readthedocs.io/en/latest/api.html
from twitter_keys import * # this file should contain your access keys to Twitter
# ------------------------------------------------------------ #
# Use the authentication keys and tokens
# ------------------------------------------------------------ #
twitter = Twython(API_key, API_key_secret, Access_Token, Access_Token_secret)
twitter.verify_credentials()
#
# ------------------------------------------------------------ #
# Get information about a Twitter user
# ------------------------------------------------------------ #
user = twitter.show_user(screen_name='londonu')
user['screen_name']
user['followers_count'] #  number of people following user
user['friends_count'] #  number of people user is following
user['statuses_count'] # number of tweets
#
most_recent_tweet = user['status']
most_recent_tweet['text'] # text of most recent tweet
#
# ------------------------------------------------------------ #
# Using Twitter's search API to get tweets of interest
# ------------------------------------------------------------ #
# Search operators:
# https://developer.twitter.com/en/docs/twitter-api/v1/tweets/search/guides/standard-operators
# https://developer.twitter.com/en/docs/twitter-api/v1/tweets/search/api-reference/get-search-tweets
#
# result_type = {'recent', 'popular', 'mixed'}
# mixed : include both popular and real time results in the response.
# recent : return only the most recent results in the response
# popular : return only the most popular results in the response.
# tweet_mode = {'compat', 'extended'}
# maximum count per request = 100
search_response = twitter.search(q='#ClimateChange', count = 100, lang = 'en', tweet_mode='extended', result_type = 'mixed')
search_tweets = search_response['statuses']
#
len(search_tweets)
#
# API method rate limits: rate limit imposed by Twitter
# limit on the number of function calls per 15-minute window
# https://developer.twitter.com/en/docs/twitter-api/v1/rate-limits
twitter.get_application_rate_limit_status()['resources']['search']
# 180 requests x 100 tweets per request in 15 min = 18000 tweets / 15 min
#
# Get *one* tweet:
tweet = search_tweets[3]
# https://developer.twitter.com/en/docs/twitter-api/v1/data-dictionary/object-model/tweet
#
# Clearer print of a tweet as a dictionary
print(json.dumps(tweet, sort_keys=False, indent=2, ensure_ascii=False))
#
tweet['id_str'] # ID string of the tweet
tweet['user']['screen_name'] # who is tweeting?
tweet['created_at'] # when was it created?
is_retweet = 'retweeted_status' in tweet # whether the tweet is a retweet
#
# (I suggest to work in 'extended' mode to avoid missing information)
compat_mode = 'text' in tweet # if we work in 'compatibility' mode
ext_mode = 'full_text' in tweet # if we work in 'extended' mode
#
if not is_retweet: # if the tweet is not a retweet
    tweet['full_text'] # what is the tweet?
    tweet['retweet_count'] # how many times the original tweet has been retweeted?
    tweet['favorite_count'] # how many times this Tweet has been liked
    tweet['metadata']['result_type'] # is it a popular or recent tweet
    # hashtags mentioned in the tweet
    l_htags = [hashtag['text'] for hashtag in tweet['entities']['hashtags']]
    len(l_htags) # if zero, no hashtag is mentioned in tweet
else: # if the tweet is a retweet
    tweet['retweeted_status']['id_str'] # ID string of the original tweet
    tweet['retweeted_status']['user']['screen_name'] # # user tweeting the original tweet
    tweet['retweeted_status']['created_at'] # when was the original tweet created?
    tweet['retweeted_status']['full_text'] # what is the full text of the original tweet?
    tweet['retweeted_status']['retweet_count'] # how many times the original tweet has been retweeted?
    tweet['retweeted_status']['favorite_count'] # how many times the original Tweet has been liked?
    tweet['retweeted_status']['metadata']['result_type'] # is it a popular or recent tweet?
    # tweet['entities']['hashtags']  # this is incomplete, because the full_text of a retweeted text is truncated, despite using the extended mode
    # complete list of hashtags of original tweet:
    l_htags = [hashtag['text'] for hashtag in tweet['retweeted_status']['entities']['hashtags']]
#
# ------------------------------------------------------------ #
# Iterating to get more than 100 tweets from a search
# ------------------------------------------------------------ #
max_tweets = 5000
search_tweets = []
last_id = -1
while len(search_tweets) < max_tweets:
    new_tweets = twitter.search(q='#ClimateChange', tweet_mode='extended', count=100, lang = 'en', result_type = 'mixed', max_id=str(last_id - 1))['statuses']
    if not new_tweets:
        break
    search_tweets.extend(new_tweets)
    last_id = new_tweets[-1]['id']
#
len(search_tweets)
#
tw_ext_mode = 0
for tw in search_tweets:
    if ('full_text' in tw):
        tw_ext_mode += 1
tw_ext_mode/len(search_tweets) # proportion of tweets in 'extended' mode
#
tw_popular = 0
for tw in search_tweets:
    if (tw['metadata']['result_type']=='popular'):
        tw_popular += 1
tw_popular/len(search_tweets) # proportion of popular tweets
#
# ------------------------------------------------------------ #
# Save results in JSON format: to explore later
# or to merge with other files for post-processing
# ------------------------------------------------------------ #
from datetime import datetime
now = datetime.now()
str_now = now.strftime("%Y_%m_%d_%H_%M_%S")
with open(str_path + '/tweets_results_' + str_now + '.json', 'w') as fh:
    json.dump(search_tweets, fh)
#
# ------------------------------------------------------------ #
# Read tweets from JSON file
# ------------------------------------------------------------ #
# Open JSON file with tweets
f = open(str_path + '/tweets_results_' + str_now + '.json')
search_tweets_json = json.load(f)
tweet = search_tweets_json[0]
#
# ------------------------------------------------------------ #
# Store tweet info in pandas dataframe(s):
# ------------------------------------------------------------ #
pd_df = pd.json_normalize(search_tweets_json)
pd_df.to_csv(str_path + '/saearch_tweets.csv') # write it as a csv file
len(pd_df.columns)
#
# select some *normalised* columns and write as csv file for inspection
pd_df[['id_str', 'created_at', 'user.screen_name', 'entities.hashtags', 'retweeted_status.user.screen_name', 'retweeted_status.entities.hashtags']].to_csv(str_path + '/saearch_tweets_hashtags.csv')
#
# ------------------------------------------------------------ #
# Work with file containing previously stored tweets
# ------------------------------------------------------------ #
search_tweets = search_tweets_json
#
# ------------------------------------------------------------ #
# Twitter retweet network
# ------------------------------------------------------------ #
#
# "retweet" -- rebroadcasting another user's tweet to your followers
retweets = []
for tweet in search_tweets:
    if 'retweeted_status' in tweet:
        retweets.append(tweet)
len(retweets) # each tweet in this list of retweets represents an edge in the retweet network
# direction of information flow: from the retweeted user to the retweeter
# a user can retweet another user more than once (ie. weighted digraph)
#
D = nx.DiGraph()
#
for retweet in retweets:
    # get tweet being retweeted
    retweeted_status = retweet['retweeted_status']
    # get user who is being retweeted (source)
    retweeted_sn = retweeted_status['user']['screen_name']
    # get user who retweets (target)
    retweeter_sn = retweet['user']['screen_name']
    # Edge direction: retweeted_sn -> retweeter_sn
    if D.has_edge(retweeted_sn, retweeter_sn):
        D.edges[retweeted_sn, retweeter_sn]['weight'] += 1
    else:
        D.add_edge(retweeted_sn, retweeter_sn, weight=1)
        D.nodes[retweeted_sn]['followers']=retweeted_status['user']['followers_count']
        D.nodes[retweeter_sn]['followers']=retweet['user']['followers_count']
#
D.number_of_edges()
D.number_of_nodes()
# D.nodes(data=True)
# D.edges(data=True)
#
# Connectivity
nx.is_weakly_connected(D) # False
nx.number_weakly_connected_components(D) # how many weakly connected components
#
# Centrality measures for directed, weighted networks
pd_C = D_centrality_pd(D)
# Most retweeted users (*influential* users)
pd_C.nlargest(10,'OutDegree', keep='all')
# Anomaly detection
# social media manipulation: accounts "spam" retweets
# detected by very high in-degree
pd_C.nlargest(10,'InDegree', keep='all')
#
# Drawing the Twitter retweet network
# Build a k-core to ease visualisation
k_level = 2
D_plot = D.copy()
D_plot.remove_edges_from(nx.selfloop_edges(D_plot))
D_plot = nx.k_core(D_plot, k_level)
# len(D_plot)
# len(D_plot.edges)
for n in D_plot.nodes():
    node_out_degree = D_plot.out_degree(n, weight='weight')
    followers = D_plot.nodes[n]['followers']
    node_size = 1
    if followers  > 0:
        node_size = int(math.log(followers))
    D_plot.nodes[n]['size'] = node_size
    # distinguish those who have been retweeted by colour
    if node_out_degree > 0:
        D_plot.nodes[n]['group'] = 1
    else:
        D_plot.nodes[n]['group'] = 2
#
plot_G_pyvis(D_plot, str_path + '/Twitter_Retweet_Network_k_' + str(k_level) + '_core.html')
#
# ------------------------------------------------------------ #
# Twitter user mention network
# ------------------------------------------------------------ #
# @userA mentions @userB in a tweet
# draw edges in the direction of attention flow
#
D = nx.DiGraph()
#
# Obtain user_mentions from 'retweeted_status' for
# retweets, in case 'user_mentions' in 'entities' has been
# truncated, missing data
for tweet in search_tweets:
    tweet_sn = tweet['user']['screen_name']
    if 'retweeted_status' in tweet: # if the tweet is a retweet
        l_sn = [tweet['retweeted_status']['user']['screen_name']] + [user_mentions['screen_name'] for user_mentions in tweet['retweeted_status']['entities']['user_mentions']]
    else: # if the tweet is not a retweet
        l_sn = [user_mentions['screen_name'] for user_mentions in tweet['entities']['user_mentions']]
    #
    for mentioned_sn in l_sn:
        my_edge = (tweet_sn, mentioned_sn)
        if D.has_edge(*my_edge): #the star unwraps the tuple
            D.edges[my_edge]['weight'] += 1
        else:
            D.add_edge(*my_edge, weight=1)
#
D.number_of_nodes()
D.number_of_edges()
# D.nodes(data=True)
# D.edges(data=True)
#
# Connectivity
nx.is_weakly_connected(D) # False
nx.number_weakly_connected_components(D) # how many weakly connected components
#
# Centrality measures for directed, unweighted networks
pd_C = D_centrality_pd(D)
#
# Most mentioned users (*popular* users)
pd_C.nlargest(10,'InDegree', keep='all')
pd_C.nlargest(10,'PageRank', keep='all')
# positive correlation, but there are cases of the same InDegree with varying PageRank
pd_C.plot.scatter(x = 'InDegree', y='PageRank', logx = True, logy = True)
#
# Conversation drivers
# user mentioning many others in a conversation may be "driving" the conversation and trying to include others in the dialogue
pd_C.nlargest(10,'OutDegree', keep='all')
pd_C.nlargest(10,'Closeness', keep='all')
# positive correlation, but there are cases of the same OutDegree with varying Closeness, as well as some outliers
pd_C.plot.scatter(x = 'OutDegree', y='Closeness', logx = True, logy = True)
#
# Drawing the Twitter user mention network
# Build a k-core to ease visualisation
k_level = 5
D_plot = D.copy()
# remove those users who mention themselves
D_plot.remove_edges_from(nx.selfloop_edges(D_plot))
D_plot = nx.k_core(D_plot, k_level)
# len(D_plot)
# len(D_plot.edges)
for n in D_plot.nodes():
    node_in_degree = D_plot.in_degree(n, weight='weight')
    node_size = 5
    if node_in_degree > 0:
        node_size = int(math.log(node_in_degree**2)) + 5
    D_plot.nodes[n]['size'] = node_size
    # distinguish those who have been mentioned by colour
    if node_in_degree > 0:
        D_plot.nodes[n]['group'] = 1
    else:
        D_plot.nodes[n]['group'] = 2
#
plot_G_pyvis(D_plot, str_path + '/Twitter_User_Mention_Network_k_' + str(k_level) + '_core.html')
#
# ------------------------------------------------------------ #
# Twitter hashtag network
# ------------------------------------------------------------ #
# The projection onto hashtags of a bipartite network of
# users mentioning hashtags (from user to hashtags)
# Hashtags will be linked when they are mentioned in the same
# tweet by a user
# But we filter hashtags to include only those having at least
# a minimum threshold of users mentioning the hashtag pair
#
D = nx.DiGraph() # Create an directed graph
hashtag_nodes = [] # a list with unique hashtags
for tweet in search_tweets:
    tweet_sn = tweet['user']['screen_name']
    if 'retweeted_status' in tweet: # if the tweet is a retweet
        l_htags = [hashtag['text'].lower() for hashtag in tweet['retweeted_status']['entities']['hashtags']]
    else:
        l_htags = [hashtag['text'].lower() for hashtag in tweet['entities']['hashtags']]
    #
    for htag in l_htags:
        htag = '#' + htag
        my_edge = (tweet_sn, htag)
        D.add_edge(*my_edge)
        if not (htag in hashtag_nodes):
            hashtag_nodes.append(htag)
        D.nodes[tweet_sn]['type']='user'
        D.nodes[htag]['type']='hashtag'
#
len(D.nodes)
len(D.edges)
# D.nodes(data=True)
# D.edges(data=True)
# ------------------------------------------------------------ #
# Filtering in the bipartite network
# ------------------------------------------------------------ #
sorted(hashtag_nodes) # list of nodes representing hashtags
#
pd_C = D_centrality_pd(D)
pd_C # InDegree = 0 is a user, OutDegree = 0 is a hashtag
#
in_degree_sequence = sorted([D.in_degree(n) for n in D.nodes()])
in_degree_counts = Counter(in_degree_sequence)
#
# Keep only those hashtags that have been mentioned by more than min_U users
min_U = 10
#
hashtag_nodes_filter = list(pd_C[pd_C['InDegree']>=min_U].index)
len(hashtag_nodes_filter)
hashtag_nodes_to_del = list(pd_C[(pd_C['InDegree']<min_U) & (pd_C['InDegree']>0)].index)
# check that we are exhausting all hashtags
len(hashtag_nodes_filter) + len(hashtag_nodes_to_del) == len(hashtag_nodes)
#
# Delete all edges with hashtag_nodes_to_del:
len(D.edges)
D_copy = D.copy()
for e in D_copy.edges:
    if (e[1] in hashtag_nodes_to_del):
        D.remove_edge(e[0], e[1])
len(D.edges)
len(D.nodes)
D = nx.k_core(D, 1) # keep only nodes with at least degree = 1
# nodes that pointed *only* to eliminated hashtags will be elminated, as they became singletons (i.e. with degree 0)
len(D.nodes)
#
# Drawing the Twitter user/hashtag bipartite network
# Build a k-core to ease visualisation
k_level = 10
D_plot = D.copy()
D_plot = nx.k_core(D_plot, k_level)
len(D_plot.nodes)
len(D_plot.edges)
# Plot bipartite network
for n in D_plot.nodes():
    # distinguish node types by colour
    if D_plot.nodes[n]['type'] == 'user':
        D_plot.nodes[n]['color'] = '#3182bd' # it's a user
    else:
        D_plot.nodes[n]['color'] = '#d95f0e' # it's a hashtag
for e in D_plot.edges():
    D_plot.edges[e]['color'] = '#3182bd'
#
plot_G_pyvis(D_plot, str_path + '/Twitter_Bipartite_User_Hashtag_Network_k_' + str(k_level) + '_core.html')
#
# ------------------------------------------------------------ #
# Project bipartite network into Hashtag co-occurrence network
# ------------------------------------------------------------ #
# weights represent number of shared neighbors in the bipartite network, i.e. number of users mentioning both hashtags in *one* tweet.
# Note: there may be links with weight = 1. In those cases, only one user is mentioning two hashtags *together*, despite each hashtag being mentioned several times *separately*.
from networkx.algorithms import bipartite
G_htag = bipartite.weighted_projected_graph(D.to_undirected(), hashtag_nodes_filter, ratio=False)
# len(G_htag.nodes) == len(hashtag_nodes_filter)
#
# ------------------------------------------------------------ #
# Filtering in the projected network
# ------------------------------------------------------------ #
# We may delete all links with weight lower than a given threshold (e.g. at least min_U users mentioning the hashtags together):
#
# Have a look at the degree distribution:
degree_sequence = sorted([G_htag.degree(n) for n in G_htag.nodes()])
degree_counts = Counter(degree_sequence)
#
min_U = 10
len(G_htag.nodes)
len(G_htag.edges)
G_htag_copy = G_htag.copy()
for e in G_htag_copy.edges(data=True):
    if(e[2]['weight']<min_U):
        G_htag.remove_edge(e[0], e[1])
#
len(G_htag.edges)
G_htag = nx.k_core(G_htag, 1) # keep only nodes with at least degree = 1
len(G_htag.nodes)
#
# Centrality measures for undirected, unweighted networks
pd_C = G_centrality_pd(G_htag)
#
pd_C.nlargest(20,'Degree', keep='all')
pd_C.nlargest(20,'Closeness', keep='all')
pd_C.nlargest(20,'Betweenness', keep='all')
#
# Drawing the Twitter hashtag network
# Build a k-core to ease visualisation
k_level = 4
G_plot = G_htag.copy()
G_plot = nx.k_core(G_plot, k_level)
len(G_plot)
#
plot_G_pyvis(G_plot, str_path + '/Twitter_Hashtag_Network_k_' + str(k_level) + '_core.html', bln_weighted=True)
#
# EOF
