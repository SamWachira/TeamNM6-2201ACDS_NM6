"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""



# Streamlit dependencies
import streamlit as st
import joblib,os
from PIL import Image
import re
import string
import html
import nltk

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

from nltk.tokenize import word_tokenize, TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.tag.perceptron import PerceptronTagger
from nltk.corpus import stopwords, wordnet 


# Data dependencies
import pandas as pd
from IPython.display import Markdown


# Load your raw data
raw = pd.read_csv("../resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """
	# Creates a main title and subheader on your page -
	# these are static across all pages
	#edsa_logo_image = Image.open('../resources/imgs/EDSA_logo.png')
	#st.image(image=edsa_logo_image)



	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Home Page",	"This APP", "Information", "Exploratory Data Analysis", "Prediction"]
	selection = st.sidebar.selectbox("Choose Option", options)


	# Landing Page
	if selection == "Home Page":
		# Adds the start up business logo
		business_logo_image = Image.open('../resources/imgs/Business_Logo.jpg')
		team_image = Image.open('../resources/imgs/team.png')
		about_us_image = Image.open('../resources/imgs/about_us.png')
		
		# Company name and Logo
		col_1, mid, col_2 = st.columns([1, 5, 20])
		with col_1:
			st.image(image=business_logo_image, caption="Business Logo", width=120)

		with col_2:
			st.header("*EMERALD*")


		# About the comapny
		st.markdown("##")
		st.markdown("##")
		about_company = "_*EMERALD is a Data Science Start up Company that helps companies create effective \
						marketing tools using machine learning in order to increase advertising efficiency \
						by helping them have a better understanding of their target market*_."
		st.markdown(about_company)


		# Adding spaces and a separator
		st.markdown("##")	# Adds spaces
		st.markdown("##")	# Adds spaces
		st.markdown("***")	# Adds a separator
		
		# Team members
		team_name = "The Team"
		st.subheader(team_name)
		st.markdown("##")	# Adds space
		st.image(image=team_image, caption="Members of the Team")

		# Adding spaces and a separator
		st.markdown("##")	# Adds spaces
		st.markdown("##")	# Adds spaces
		st.markdown("***")	# Adds a separator

		# About Us
		team_name = "About Us"
		st.subheader(team_name)
		st.markdown("##")
		st.image(image=about_us_image, caption="About Us.")


	# About the app
	if selection == "This APP":

		# App logo
		ml_logo_image = Image.open('../resources/imgs/ml_logo.png') # this opens the logo image

		col_1, mid, col_2 = st.columns([1, 10, 20])
		with col_1:
			st.image(image=ml_logo_image, caption="Planet Solver", width=180)

		with col_2:
			st.header("**_PLANET_ SOLVER**")

		# About the web app
		st.markdown("##")
		st.markdown("##")
		planet_solver = "**_PLANET_ SOLVER** is a Machine Learning Model that helps companies \
						to identify whether or not a person believes in climate change based on \
						their tweet and could possibly be converted to a new customer."
		st.markdown(planet_solver)



	# Building out the "Information" page
	if selection == "Information":
		st.title("Tweet Classifer")
		st.subheader("Climate change tweet classification")		
		st.info("General Information")
		
		# load the markdown containing information about the project.
		with open('../resources/info.md', 'r') as f:
			text = f.read()

		# You can read a markdown file from supporting resources folder
		st.markdown(text)

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page


	# Building and displaying the Exploratory data analysis page
	if selection == "Exploratory Data Analysis":

		st.title("Exploratory Data Analysis")

		# Selecting which graphical illlustration to view
		eda_option = ["Overview" , "Tweet Distribution", "Length of tweet per class", "Top hastags per class"]
		eda_labels = "Select a graphical distribution of tweets to view."

		eda_selection = st.sidebar.radio(eda_labels, eda_option)

		# Overview of the EDA
		if eda_selection == "Overview":
			st.subheader("Overview")
			eda_text = "This section shows some graphical illustrations of the statistical distribution of tweets \
					in the dataset used in training the Machine Learning Models.\
					\n\nThree major plots were used during the exploratory data analysis of the train data set, namely:\
					\n1. The distribution of the data set.\
					\n2. The length of Tweets for each category\
					\n3. And the top hashtags for each category\
					\n\n This will later be seen to be an insight into how convinced each category of the people are about their opinions."
			st.markdown(eda_text)

		# Tweet Distribution
		if eda_selection == "Tweet Distribution":

			st.subheader("Tweet Distribution")
			text1 = "The image below shows a plot depicting how the tweets are distributed across each class."
			st.markdown(text1)
			tweet_dist_image = Image.open('../resources/plot_images/tweet_distn.png')
			st.image(image=tweet_dist_image, caption="Tweet distribution")
			text2 = "A close look at the above distribution indicates that the data is severely imbalanced with the majority \
			of tweets falling in the 'pro' category, supporting the belief of man-made climate change while just 6% are anti-climate change."
			st.markdown(text2)

		# Length of Tweet per class
		if eda_selection == "Length of tweet per class":
			st.subheader("Tweet per Class")
			length_of_tweet_per_class_image = Image.open('../resources/plot_images/length_of_tweet_per_class.png')
			st.image(image=length_of_tweet_per_class_image, caption="Box plot for Length of tweet per class")

			text = "From the boxplot, it could be observed that tweets that fall in the pro climate change class are\
			 generally longer and the shortest tweets belong to the anti climate change class.\
			 \n\nThis can be seen from the respective spans of the plots.\
				\n\nIt seems that we have strong-witted pros here!! \n\nIt is worthy to also not that\
				 neutral climate change tweets tend to have the most variability in tweet length."
			st.markdown(text)

		# Top hashtags
		if eda_selection == "Top hastags per class":
			st.subheader("Top Hashtags")
			text1 = "Hashtags being a powerful feature used in sorting and organizing tweets \
			 provide an excellent approach to show that a content is related to a specific issue.\
			 \n\nIt could be helpful in unraveling what the most popular hashtags are in each of the classes which would\
			help in obtaining a better grasp of the types of knowledge ingested and shared by members of each class."
			st.markdown(text1)

			st.text("\n\n\n\n\n")
			# Graph 1 for hashtags
			image_text_1 = "Top Hashtags for Pro Climate Change"
			st.markdown(image_text_1)
			pro_image = Image.open('../resources/plot_images/top_hashtags_pro_climate_change.png')
			st.image(image=pro_image, caption="Top Hashtags for Pro Climate Change")

			st.text("\n\n\n\n\n")
			# Graph 2 for hashtags
			image_text_2 = "Top Hashtags for Anti Climate Change"
			st.markdown(image_text_2)
			anti_image = Image.open('../resources/plot_images/top_hashtags_anti_climate_change.png')
			st.image(image=anti_image, caption="Top Hashtags for Anti Climate Change")

			st.text("\n\n\n\n\n")
			# Graph 3 for hashtags
			image_text_3 = "Top Hashtags for News on Climate Change"
			st.markdown(image_text_3)			
			news_image = Image.open('../resources/plot_images/top_hashtags_news_class.png')
			st.image(image=news_image, caption="Top Hashtags for News on Climate Change")

			st.text("\n\n\n\n\n")
			# Graph 4 for hashtags
			image_text_4 = "Top Hashtags for Neutral tweets on Climate Change"
			st.markdown(image_text_4)			
			neutral_image = Image.open('../resources/plot_images/top_hashtags_neutral_class.png')
			st.image(image=neutral_image, caption="Top Hashtags for Neutral tweets on Climate Change")									

			text2 = "Knowing the popular words used across various classes can help to understand how customers think or are thinking."
			st.markdown(text2)






	# Building out the prediction page
	if selection == "Prediction":
		st.title("Tweet Classification using Machine Learning Models")
		#st.subheader("Climate change tweet classification")	
		#st.subheader("Modelling")		
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")


		# cleaning the input text before passing it into the machine learning model for prediction
		tweet_text = tweet_text.replace('\n', '')
		tweet_text = html.unescape(tweet_text)

		# removing special characters from text
		tweet_text = re.sub(r"(@[A-Za-z0-9_]+)|[^\w\s]|http\S+", "", tweet_text)

		tweet_text = tweet_text.lower()		# converts text to lowercase
		tokeniser = TreebankWordTokenizer()		# creating an instance of the TreebankWordTokenizer
		tweet_text_token = tokeniser.tokenize(tweet_text)		# transforming input into tokens

		#This function obtains a pos tag and returns the part of speech.
			#Input:
			#	tag: POS tag
			#	datatype: str
			#Output:
			#	wordnet.pos: Part of Speech
			#	datatype: str
		

		def get_pos(tag):
			if tag.startswith('V'):
				return wordnet.VERB

			elif tag.startswith('J'):
				return wordnet.ADJ

			elif tag.startswith('R'):
				return wordnet.ADV
			
			elif tag.startswith('N'):
				return wordnet.NOUN
			
			else:
				return wordnet.NOUN

		pos_tag = PerceptronTagger()
		tweet_text_POS_tag = pos_tag.tag(tweet_text_token)		# gets the part of speech tag of each word in the sentence

		tweet_text_POS_tag = [(word, get_pos(tag)) for (word, tag) in tweet_text_POS_tag]

		lemmatizer = WordNetLemmatizer()		# creating an instance of the lemmatizer
		
		tweet_text_token = [lemmatizer.lemmatize(token) for token in tweet_text_token]		# gets the lemma of each word

		tweet_text_token =  [word for word in tweet_text_token if not word in stopwords.words('english') and word != 'not']

		tweet_text_token = ' '.join(tweet_text_token)


		model_option = ["Logistic Regression", "Linear Support Vector Classifier", "Random Forest Classifier", "Naive Bayes",
							"K Nearest Neighbour Classifier", "Linear SVC using Optimal Hyperparameters"]	# A list of available models that can be used for the classification

		model_selection = st.selectbox("Select a model type you will like to use as the classifier.", model_option)

			# Selecting from multiple models
			# If Logistic Regression is selected
		if model_selection == "Logistic Regression":
			predictor = joblib.load(open(os.path.join("../resources/logistic_regression_model.pkl"),"rb"))

			# If Linear Support Vector Classifier is selected
		if model_selection == "Linear Support Vector Classifier":
			predictor = joblib.load(open(os.path.join("../resources/linear_svc_model.pkl"),"rb"))

			# If Random Forest Classifier is selected
		if model_selection == "Random Forest Classifier":
			predictor = joblib.load(open(os.path.join("../resources/random_forest_model.pkl"),"rb"))

			# If Naive Bayes is selected
		if model_selection == "Naive Bayes":
			predictor = joblib.load(open(os.path.join("../resources/naive_bayes_model.pkl"),"rb"))

			# If K Nearest Neighbour Classifier is selected
		if model_selection == "K Nearest Neighbour Classifier":
			predictor = joblib.load(open(os.path.join("../resources/knn_model.pkl"),"rb"))

			# If Linear SVC using Optimal Hyperparameters is selected
		if model_selection == "Linear SVC using Optimal Hyperparameters":
			predictor = joblib.load(open(os.path.join("../resources/lsvc_op_model.pkl"),"rb"))

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = [tweet_text]

			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			
			prediction = predictor.predict([tweet_text])

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output more human interpretable.

			# A dictionary to show a more human interpretation of result

			result_dict = {'News': 'This tweet links to factual news about climate change.',
							'Pro': 'This tweet supports the belief of man-made climate change.',
							'Neutral': 'This tweet neither supports nor refutes the belief of man-made climate change.',
							'Anti': 'The tweet does not believe in man-made climate change.'
			}
			result = result_dict[prediction[0]]
			st.success("Text Categorized as: {}".format(prediction[0]))
			st.success("{}".format(result))



# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
