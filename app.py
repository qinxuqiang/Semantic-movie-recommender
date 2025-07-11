#imports
import spaces
import pandas as pd
import numpy as np
import json
import torch
import re
import warnings
warnings.filterwarnings('ignore')

#from langchain_community.document_loaders import TextLoader
#from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings



# Auto-detect device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")



#data
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",    
                                   model_kwargs={'device': 'cuda'},    
                                   encode_kwargs={'normalize_embeddings': True})
vectorstore=FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)
movies=pd.read_csv('movies_rec_db.csv')

json_columns=['keywords_cleaned',
              'production_countries_cleaned',
              'spoken_languages_cleaned',
              'cast',
              'directors']
for col in json_columns:
  movies[col]=movies[col].apply(json.loads)



director_list_list=movies['directors'].to_list()
director_list=[]
for i in director_list_list:
  for j in i:
    director_list.append(j)


cast_list_list=movies['cast'].to_list()
cast_list=[]
for i in cast_list_list:
  for j in i:
    cast_list.append(j)

genre_list=['Action','Adventure','Animation','Comedy','Crime','Documentary','Drama'
'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Mystery', 'Romance',
        'Science Fiction', 'TV Movie', 'Thriller', 'War', 'Western']

year_list=['1910s','1920s','1930s','1940s','1950s','1960s','1970s','1980s','1990s','2000s','2010s','2020s']
year_list.reverse()



#fuzzy search
from rapidfuzz import fuzz


# Nickname dictionary
NICKNAME_MAP = {
    "abby": ["abigail"], "ali": ["alison", "alice"], "ally": ["alison", "alice"],
    "andy": ["andrew"], "barb": ["barbara"], "beccy": ["rebecca"], "becky": ["rebecca"],
    "ben": ["benjamin"], "beth": ["elizabeth"], "bill": ["william"], "bob": ["robert"],
    "carrie": ["caroline", "carol"], "cathy": ["catherine", "katherine"],
    "charlie": ["charles", "charlotte"], "chuck": ["charles"], "danny": ["daniel"],
    "dan": ["daniel"], "dave": ["david"], "dick": ["richard"], "ed": ["edward", "edmund"],
    "eddie": ["edward", "edmund"], "frank": ["francis", "franklin"], "grace": ["gracie"],
    "hank": ["henry"], "harry": ["harold", "henry"], "jack": ["john"],
    "jackie": ["jacqueline"], "jake": ["jacob"], "jen": ["jennifer"],
    "jenny": ["jennifer"], "jerry": ["jerome", "gerald"], "jim": ["james"],
    "joey": ["joseph"], "joe": ["joseph"], "john": ["jonathan", "jon"],
    "jon": ["jonathan", "john"], "kate": ["katherine", "catherine"],
    "kathy": ["katherine", "catherine"], "larry": ["lawrence"], "leo": ["leonard", "leonardo"],
    "liz": ["elizabeth"], "luke": ["lucas", "lucius"], "maggie": ["margaret"],
    "mandy": ["amanda"], "marge": ["margaret"], "matt": ["matthew"], "meg": ["margaret"],
    "mike": ["michael"], "nancy": ["anne", "ann"], "nate": ["nathan", "nathaniel"],
    "nick": ["nicholas"], "pat": ["patrick", "patricia"], "peggy": ["margaret"],
    "pete": ["peter"], "rick": ["richard"], "rich": ["richard"], "rob": ["robert"],
    "ron": ["ronald"], "ronnie": ["ronald"], "sally": ["sarah"], "sam": ["samuel", "samantha"],
    "steve": ["steven", "stephen"], "sue": ["susan", "suzanne"], "suzie": ["susan", "suzanne"],
    "ted": ["edward", "theodore"], "tina": ["christina", "christine"], "tom": ["thomas"],
    "tony": ["anthony"], "trish": ["patricia"], "vicky": ["victoria"], "zack": ["zachary"],
}


def normalize(s):
    """Lowercase and remove non-alphanumeric characters"""
    return re.sub(r'\W+', '', s).lower()

def fuzzy_name_search(query, name_list, nickname_map=NICKNAME_MAP, threshold=50, limit=5):
    """Fuzzy search with nickname expansion and normalization.

    Args:
        query (str): Input name (possibly inaccurate).
        name_list (List[str]): List of full names to search.
        nickname_map (Dict[str, List[str]]): Optional nickname mappings.
        threshold (int): Minimum score to keep a match.
        limit (int): Max number of results to return.

    Returns:
        List[Tuple[str, float]]: Top matching names and scores.
    """
    # Normalize names and store mapping
    normalized_names = {normalize(name): name for name in name_list}
    normalized_query = normalize(query)

    # Expand query with nickname variants
    expanded_queries = {normalized_query}
    for nickname, full_forms in nickname_map.items():
        if nickname in normalized_query:
            for full in full_forms:
                variant = normalized_query.replace(nickname, normalize(full))
                expanded_queries.add(variant)

    # Fuzzy match each variant against the normalized names
    results = []
    for variant in expanded_queries:
        for norm_name, original_name in normalized_names.items():
            score = fuzz.ratio(variant, norm_name)
            if score >= threshold:
                results.append((original_name, score))

    # Deduplicate and sort by score
    results = sorted(set(results), key=lambda x: x[1], reverse=True)

    return results[:limit]


#MAIN SEARCH FUNCTION:

@spaces.GPU
def retrieve_semantic_recommendations(
        query: str,
        director: str="",
        cast: str = "",
        genre: str = "All",
        year: str = "All",
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:
  """
  Retrieves movie recommendations based on user input.


  Args:
      query: user's description of the movie
      director: if specified, is the user's estimation of the name of the director
      cast: if specified, is the user's estimation of the name of the cast
      genre: if specified, is the user's description of the genre of the movie
      year: if specified, is the release year of the movie
      initial_top_k: initial number of movies to retrieve from sementic search after filtering through director, cast, genre and year
      final_top_k: final number of movies to return

  Returns:
      pd.DataFrame: A dataframe of recommended movies.
  """
  movie_rec=movies.copy()

# Hard restrictions: genre and year
  if genre != "All":
        movie_rec = movie_rec[movie_rec["genres_cleaned"].apply(lambda x: genre in x)]

  if year != "All":
        year= int(year[:4])
        movie_rec = movie_rec[movie_rec["year"].apply(lambda x: x in range(year,year+10))]

# FLEXIBLE RESTRICTIONS: Director
  if director.strip():  # Check for non-empty string
      director = director.strip()
      if director in director_list:
          # Exact match found
          movie_rec = movie_rec[movie_rec["directors"].apply(lambda x: director in x)]
          print(f"After director filter (exact: {director}): {len(movie_rec)} movies")
      else:
          # Fuzzy matching
          try:
              dir_guesses = set([entry[0] for entry in fuzzy_name_search(director, director_list)[:2]])
              if dir_guesses:  # Only apply filter if matches found
                  movie_rec = movie_rec[
                      movie_rec["directors"].apply(lambda x: any(d in dir_guesses for d in x))
                  ]
                  print(f"After director filter (fuzzy: {dir_guesses}): {len(movie_rec)} movies")
              else:
                  print(f"Warning: No director matches found for '{director}'")
          except Exception as e:
              print(f"Error in director search: {e}")

# FLEXIBLE RESTRICTIONS: Cast
  if cast.strip():  # Check for non-empty string
      cast = cast.strip()
      if cast in cast_list:
          # Exact match found
          movie_rec = movie_rec[movie_rec["cast"].apply(lambda x: cast in x)]
          print(f"After cast filter (exact: {cast}): {len(movie_rec)} movies")
      else:
          # Fuzzy matching
          try:
              cast_guesses = set([entry[0] for entry in fuzzy_name_search(cast, cast_list)[:2]])
              if cast_guesses:  # Only apply filter if matches found
                  movie_rec = movie_rec[
                      movie_rec["cast"].apply(lambda x: any(c in cast_guesses for c in x))
                  ]
                  print(f"After cast filter (fuzzy: {cast_guesses}): {len(movie_rec)} movies")
              else:
                  print(f"Warning: No cast matches found for '{cast}'")
          except Exception as e:
              print(f"Error in cast search: {e}")
    
# Soft restrictions: querry
  if query:
    recs = vectorstore.similarity_search(query, k=initial_top_k)
    movie_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    movie_rec = movie_rec[movie_rec["id"].isin(movie_list)].head(initial_top_k)



  movie_rec = movie_rec.head(final_top_k)

  return movie_rec

def recommend_movies(
        query: str,
        director: str,
        cast: str ,
        genre: str,
        year: str,
):
    recommendations = retrieve_semantic_recommendations(query,director,cast,genre,year)

    results = []

    for _, row in recommendations.iterrows():

        description = row["overview"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:20]) + "..."
        '''

        caption = f"{row['title']}\n\
        {row['year']}, {row['runtime']} \n\
        Director: {', '.join(row['directors'])}\n\
        Cast: {', '.join(row['cast'])}\n{truncated_description}"
        '''
        caption = f"{row['title']} by {', '.join(row['directors'])}: {truncated_description}"
        results.append((row["thumbnail_url"], caption))


    return results

genre_list_aug=['All']+genre_list
year_list_aug=['All']+year_list


temp=movies.sort_values(by='popularity', ascending=False).head(16)
initial=[]
for _,row in temp.iterrows():
  description = row["overview"]
  truncated_desc_split = description.split()
  truncated_description = " ".join(truncated_desc_split[:20]) + "..."
  caption = f"{row['title']} by {', '.join(row['directors'])}: {truncated_description}"
  initial.append((row['thumbnail_url'],caption))


#Gradio Interface:
import gradio as gr

with gr.Blocks(theme = gr.themes.Monochrome()) as dashboard:
    gr.Markdown("# Semantic movie recommender")
    gr.Markdown("Find movies based on themes, directors, actors, and preferences")

    with gr.Row():
        with gr.Column(scale=2):
            user_query = gr.Textbox(
                label="Movie Description", 
                placeholder="e.g., A movie about forgiveness, time travel, or redemption",
                lines=2
            )
        with gr.Column(scale=1):
            director_query = gr.Textbox(
                label="Director", 
                placeholder="e.g., Quentin Tarantino"
            )
        with gr.Column(scale=1):
            cast_query = gr.Textbox(
                label="Actor/Actress", 
                placeholder="e.g., Keanu Reeves"
            )
    with gr.Row():
        genre_dropdown = gr.Dropdown(
            choices=genre_list_aug, 
            label="Genre", 
            value="All"
        )
        year_dropdown = gr.Dropdown(
            choices=year_list_aug, 
            label="Release Period", 
            value="All"
        )
        submit_button = gr.Button("🔍 Find Movies", variant="primary")

    gr.Markdown("## 🎭 Recommendations")
    output = gr.Gallery(
        label="Recommended Movies", 
        columns=4,  # Reduced for better mobile view
        rows=2,
        height="auto",
        show_label=True,
        value=initial
    )

    #gr.Markdown("## You might also like")

    #output2 = gr.Gallery(label = "Related movies", columns = 8, rows = 1)

    submit_button.click(fn = recommend_movies,
                        inputs = [user_query, director_query,cast_query,genre_dropdown,year_dropdown],
                        outputs = output)


    
dashboard.launch()
