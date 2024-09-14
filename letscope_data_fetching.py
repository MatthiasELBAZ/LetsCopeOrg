# %%
import sqlalchemy as db
from sqlalchemy import text
import pandas as pd
import os
import json

# %% [markdown]
# # SQLALchemy DB Connection

# %%
# db connect params
db_user_name = os.environ.get('DB_USER_NAME')
db_password = os.environ.get('DB_PASSWORD')
db_host = os.environ.get('DB_HOST')
db_name = os.environ.get('DB_NAME')
db_dialect = os.environ.get('DB_DIALECT')
db_driver = os.environ.get('DB_DRIVER')

# create engine string
engine_string = f'{db_dialect}+{db_driver}://{db_user_name}:{db_password}@{db_host}/{db_name}'

# create engine
engine = db.create_engine(engine_string)

# create connection
connection = engine.connect()

# create metadata
metadata = db.MetaData()

# %% [markdown]
# # All Tables

# %%
# print all table names
query = "show tables"
all_tables = pd.read_sql_query(text(query), con=connection)
print(all_tables)

# %% [markdown]
# # Content Data - DailyCope Videos

# %% [markdown]
# ## Programs

# %%
query = "SELECT * FROM cope_prod.dev_cope_programs"
df_dev_cope_programs = pd.read_sql(text(query), con=connection)

query = "SELECT * FROM cope_prod.dev_cope_content"
df_dev_cope_content = pd.read_sql(text(query), con=connection)

# %% [markdown]
# ## Videos

# %%

# Define a function to parse the JSON in 'dev_cope_program_param' and extract video details
def extract_videos(row):
    program_details = json.loads(row['dev_cope_program_param'])
    videos = []
    if 'videos' in program_details:  # Assuming the key for the list of videos is 'videos'
        for video in program_details['videos']:
            videos.append({
                'program_id': row['dev_cope_program_id'],
                'program_author': program_details.get('author', ''),
                'program_title': program_details.get('title', ''),
                'program_description': program_details.get('description', ''),
                'video_id': video.get('video_id', ''),
                'video_needs': video.get('needs', ''),
                'video_title': video.get('title', ''),
                'video_description': video.get('description', ''),
                'video_url': video.get('video_url', '')
            })
    return videos

# Apply the function to each row in the DataFrame and create a list of all videos
all_videos = df_dev_cope_programs.apply(extract_videos, axis=1).sum()  # Flatten the list of lists

# Create a new DataFrame from the list of video details
video_df = pd.DataFrame(all_videos)

video_df['program_author'] = video_df['program_author'].apply(lambda x: 'unknown' if x=='' else x)
video_df['video_url'] = video_df['video_url'].apply(lambda x: 'unknown' if x=='' else x)
video_df['video_description'] = video_df['video_description'].apply(lambda x: 'unknown' if x=='' or x=='N/A' else x)
video_df['video_title'] = video_df['video_title'].apply(lambda x: 'unknown' if x=='title' else x)
video_df['video_needs'] = video_df['video_needs'].apply(lambda x: 'unknown' if x=='' or x=='needs' else x)


df_dev_cope_content['video_id'] = df_dev_cope_content['dev_cope_content_content'].apply(lambda x: json.loads(x)['video_id'] if 'video_id' in json.loads(x) else '0')

# df_dev_cope_content['video_id'] = df_dev_cope_content['video_id'].astype(int)
# video_id['video_id'] = video_df['video_id'].astype(int)


video_df = pd.merge(video_df, df_dev_cope_content[['dev_cope_content_id', 'video_id']], on='video_id', how='left')

# %%
# Find video_ids in content_df that are not in video_df
unique_video_ids = set(df_dev_cope_content['video_id']) - set(video_df['video_id'])

# Filter content_df for these unique videos
unique_videos_df = df_dev_cope_content[df_dev_cope_content['video_id'].isin(unique_video_ids)]

# Extract necessary fields from the 'dev_cope_content_content' JSON and create the new DataFrame
def parse_video_details(row):
    video_details = json.loads(row['dev_cope_content_content'])
    return {
        'program_id': 'unknown',  # Unknown program ID
        'program_author': video_details.get('autor', ''),  # Use video author as program author
        'program_title': 'unknown',  # Unknown program title
        'program_description': 'unknown',  # Unknown program description
        'video_id': video_details.get('video_id', ''),
        'video_needs': 'unknown',  # Assume no specific needs provided
        'video_title': video_details.get('video_title', ''),
        'video_description': video_details.get('video_description', ''),
        'video_url': video_details.get('video_url', ''),
        'dev_cope_content_id': row['dev_cope_content_id']
    }

# Apply the function to extract video details
additional_videos_df = unique_videos_df.apply(parse_video_details, axis=1, result_type='expand')


# Append these new rows to the merged_df
video_df = pd.concat([video_df, additional_videos_df], ignore_index=True)

# %%
video_df.info()

# %%
video_df.to_csv('./data/data_content/data_program_video_content.csv', index=False)

# %% [markdown]
# # Quizes

# %%
query = "SELECT * FROM cope_prod.dev_cope_quiz_questions"
df_dev_cope_quiz_questions = pd.read_sql(text(query), con = connection)

query = "SELECT * FROM cope_prod.dev_cope_quizes"
df_dev_cope_quizes = pd.read_sql(text(query), con = connection)

# %%
quiz_dict = {}
for i in df_dev_cope_quiz_questions.dev_cope_quiz_id.unique():
    quiz_data = df_dev_cope_quiz_questions[df_dev_cope_quiz_questions.dev_cope_quiz_id==i].dev_cope_quiz_question_parameter.tolist()

    quiz_data = [json.loads(x) for x in quiz_data]

    title = json.loads(df_dev_cope_quizes[df_dev_cope_quizes.dev_cope_quiz_id==i].dev_cope_quiz_params.values[0])['title']
    description = json.loads(df_dev_cope_quizes[df_dev_cope_quizes.dev_cope_quiz_id==i].dev_cope_quiz_params.values[0])['description']

    quiz_dict[str(i)] = {'title': title, 'description': description, 'quiz': quiz_data}


# %%
with open('./data/data_quiz/all_quizes_data.json', 'w') as f:
    json.dump(quiz_dict, f)

# %% [markdown]
# # Membership

# %%
query = "SELECT * FROM cope_prod.dev_cope_membership"
df_dev_cope_membership = pd.read_sql(text(query), con = connection)[['dev_cope_membership_id',	'dev_cope_membership_type',	'dev_cope_membership_params']]
dict_membership = df_dev_cope_membership.to_dict(orient = 'index')


# %%
with open('./data/data_membership/dict_membership.json', 'w') as f:
    json.dump(dict_membership, f)

# %% [markdown]
# # User

# %%
query = "SELECT * FROM cope_prod.dev_cope_customers"
df_dev_cope_customers = pd.read_sql(text(query), con = connection)[['dev_cope_customer_uid', 'dev_cope_customer_status', 'dev_cope_email_validado', 'dev_cope_customer_active_subscription']]

query = "SELECT * FROM cope_prod.dev_cope_quiz_submit"
df_dev_cope_quiz_submit = pd.read_sql(text(query), con = connection)[['dev_cope_customer_uid', 'dev_cope_quiz_submit_quiz_id', 'dev_cope_quiz_submit_data', 'dev_cope_quiz_submit_score', 'dev_cope_quiz_submit_date']]

query = "SELECT * FROM cope_prod.dev_cope_membership_trx"
df_dev_cope_membership_trx = pd.read_sql(text(query), con = connection)[['dev_cope_customer_uid', 'dev_cope_membership_id', 'dev_cope_membership_trx_info', 'dev_cope_membership_trx_start', 'dev_cope_membership_trx_next']]

query = "SELECT * FROM cope_prod.dev_cope_watchlist_trx"
df_dev_cope_watchlist_trx = pd.read_sql(text(query), con = connection)[['dev_cope_customer_uid', 'dev_cope_content_id']]


# %%
user_uid = "r3h5KzAFj1N0LDqW8aJciUuDRBH2"

# %% [markdown]
# ## Quiz Submitted

# %%
df_user_last_quiz_submitted = df_dev_cope_quiz_submit[df_dev_cope_quiz_submit.dev_cope_customer_uid == user_uid].sort_values(by='dev_cope_quiz_submit_date', ascending = True).drop_duplicates(subset = ['dev_cope_quiz_submit_quiz_id'], keep = 'last')
df_user_last_quiz_submitted['dev_cope_quiz_submit_date'] = df_user_last_quiz_submitted['dev_cope_quiz_submit_date'].astype(str)

user_submitted_quiz = df_user_last_quiz_submitted.dev_cope_quiz_submit_data.tolist()

user_quiz = {'user_uid': user_uid, 'quiz_submitted': []}
for idx, row in df_user_last_quiz_submitted.iterrows():
    quiz_id = row.dev_cope_quiz_submit_quiz_id
    quiz_submitted = eval(row.dev_cope_quiz_submit_data)
    quiz_data = quiz_dict[str(quiz_id)]['quiz']
    description = quiz_dict[str(quiz_id)]['description']
    title = quiz_dict[str(quiz_id)]['title']
    score = row.dev_cope_quiz_submit_score
    date = row.dev_cope_quiz_submit_date

    user_quiz['quiz_submitted'].append({'quiz_id': quiz_id, 'title': title, 'description': description, 'score': score, 'date': date, 'quiz_qa': []})

    for i in range(len(quiz_submitted)):
        question = quiz_data[i]['question']
        answer = quiz_data[i]['response_options'][int(quiz_submitted[i]['value'])]['value']
        
        user_quiz['quiz_submitted'][-1]['quiz_qa'].append({'question': question, 'answer': answer})

# %% [markdown]
# ## Content Consummed

# %%
df_user_content_watched = df_dev_cope_watchlist_trx[df_dev_cope_watchlist_trx.dev_cope_customer_uid == user_uid]

user_content = {'user_uid': user_uid, 'content_watched': []}
for idx, row in df_user_content_watched.iterrows():
    content_id = row.dev_cope_content_id
    content = video_df[video_df.dev_cope_content_id == content_id]

    author = content['program_author'].values[0]
    title = content['video_title'].values[0]
    description = content['video_description'].values[0]
    video_url = content['video_url'].values[0]
    user_content['content_watched'].append({'content_id': content_id, 'video_title': title, 'video_description': description, 'author': author, 'video_url': video_url, 'program_description': content['program_description'].values[0],})

# %% [markdown]
# ## Membership Paid

# %%
df_user_membership = df_dev_cope_membership_trx[df_dev_cope_membership_trx.dev_cope_customer_uid == user_uid].reset_index(drop = True)

df_user_membership['dev_cope_membership_trx_start'] = df_user_membership['dev_cope_membership_trx_start'].astype(str)
df_user_membership['dev_cope_membership_trx_next'] = df_user_membership['dev_cope_membership_trx_next'].astype(str)

dict_user_membership = df_user_membership.to_dict(orient = 'records')

user_membership = {'user_uid': user_uid, 'membership': []}
for idx, row in df_user_membership.iterrows():
    membership_id = row.dev_cope_membership_id
    membership_info = dict_membership[membership_id]
    start = row.dev_cope_membership_trx_start
    next = row.dev_cope_membership_trx_next
    user_membership['membership'].append({'membership_id': membership_id, 'membership_info': membership_info, 'start': start, 'next': next})

# %% [markdown]
# ## User all dict

# %%
quiz_text = """User Quiz Data\n\n"""

for q in user_quiz['quiz_submitted']:
    quiz_text += f"""Title of the quiz: {q['title']}\n\nDescription of the quizz: {q['description']}\n\nUser Score of the quiz: {q['score']}\n\nDate submitted: {q['date']}"""

    for qa in q['quiz_qa']:
        quiz_text += f"""\n\nquestion: {qa['question']}\nanswer: {qa['answer']}"""

    quiz_text += "\n\n"
content_text = """User Content Data\n\n"""

for c in user_content['content_watched']:
    content_text += f"""Title of the content: {c['video_title']}\nDescription of the content: {c['video_description']}\nAuthor of the content: {c['author']}\nVideo URL: {c['video_url']}"""

    content_text += "\n\n"
membership_text = """User Membership Data\n\n"""

for m in user_membership['membership']:
    membership_text += f"""Membership ID: {m['membership_id']}\nType: {m['membership_info']['dev_cope_membership_type']}\nCost: {eval(m['membership_info']['dev_cope_membership_params'])['cost']}\nPeriod: {eval(m['membership_info']['dev_cope_membership_params'])['period']}\nStart Date: {m['start']}\nNext Payment Date: {m['next']}"""

    membership_text += "\n\n"


user_data = f"""User Data\n\n{quiz_text}\n\n{content_text}\n\n{membership_text}"""

# %%
print(user_data)


