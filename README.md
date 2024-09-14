# LetsCopeOrg

First Part: letscope_data_fetching.py
* reads data from database and generates data from content/videos, quizzes, memberships.
* It also generates for one user the quiz he had submitted, the content he watched, the membership he is paying and store them.
* the data data is stored into folders into formats like csv or json.

Second Part: Letscope_chat_2.py
* it uses only LlamaIndex python package and OPENAI api key service.
* Data folders
  * data_articles: docx documents given by letscope and made by professionals
  * data_content: constains the csv file generated previously with video's data
  * data_membership: contain the json of all membership
  * data_quiz: contain the json of all quizzes
  * data_user: contains all user folder with their json generated via letscope_data_fetching.py
  * data_website: contain all html pages of the website letscope.org
* For data_content, data_articles and data_website it creates a custom in memory vector database index

