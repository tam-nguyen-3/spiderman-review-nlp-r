library(tidyverse)
library(purrr)
library(readr)
library(tm)
library(stringr)
library(tidytext)
library(lubridate)

raw_data <- read_csv("/Users/tamnguyen/KENYON/STAT 226/project3/imdb-spider-man-reviews.csv")

#Clean the Review column ----
spiderman <- raw_data %>%
  mutate(Review = tolower(Review)) %>%
  mutate(Review = gsub("[^[:alnum:]]", " ", Review)) %>%
  mutate(Review = gsub("[[:digit:]]+", " ", Review)) %>%
  mutate(Review = str_squish(Review),
         Date = dmy(Date)) %>%
  mutate(Title = tolower(Title)) %>%
  mutate(Title = gsub("[^[:alnum:]]", " ", Title)) %>%
  mutate(Title = gsub("[[:digit:]]+", " ", Title)) %>%
  mutate(Title = str_squish(Title))

#Add new words to stopwords
stop_words <- stop_words %>% 
  bind_rows(
    tibble(
      word = c("spider", "spiderman", "man", "film", "movie", "movies"),
      lexicon = "custom"
    )
  )

#DTM----
dtm.control = list(
  removePunctuation = T,
  removeNumber = T)

spiderman_dtm <- Corpus(VectorSource(spiderman$Review)) %>%
  DocumentTermMatrix()

spiderman_sentiments <- tidy(spiderman_dtm)

# 1. Sentiment Analysis----
## a. AFINN----
### get afinn total scores for each review ----
spiderman_sentiments_afinn <- spiderman_sentiments %>%
  inner_join(get_sentiments('afinn'), by=c(term='word')) %>%
  mutate(total_value = value*count,
         document = as.numeric(document)) %>%
  group_by(document) %>%
  summarize(total_sentiment = sum(total_value))

### join total afinn score for each review with the whole dataset ----
spiderman_afinn <- spiderman %>%
  mutate(document = 1:nrow(.)) %>%
  left_join(spiderman_sentiments_afinn)

### Average afinn scores for each movie ----
average_afinn_by_movie <- spiderman_afinn %>%
  group_by(Movie) %>%
  summarize(mean_afinn = mean(total_sentiment, na.rm=T)) %>%
  arrange(desc(mean_afinn))

#### visualize: average_afinn_by_movie ----
average_afinn_by_movie %>%
  ggplot(aes(mean_afinn, reorder(Movie, mean_afinn)))+
  geom_col(fill = "#29329b")+
  labs(x = "Average Afinn by Movie", y = NULL)

### Average afinn score for each actor ----
average_afinn_by_actor <- spiderman_afinn %>%
  filter(Movie != "Spider-Man: Into the Spider-Verse") %>%
  mutate(actor = case_when(
    Movie == "Spider-Man" ~ "Tobey Maguire",
    Movie == "Spider-Man 2" ~ "Tobey Maguire",
    Movie == "Spider-Man 3" ~ "Tobey Maguire",
    Movie == "The Amazing Spider-Man" ~ "Andrew Garfield",
    Movie == "The Amazing Spider-Man 2" ~ "Andrew Garfield",
    Movie == "The Amazing Spider-Man" ~ "Andrew Garfield",
    TRUE ~ "Tom Holland"
  )) %>%
  group_by(actor) %>%
  summarize(avg_afinn = mean(total_sentiment, na.rm = T)) %>%
  arrange(desc(avg_afinn))

#### visualize: avg afinn by actor----
average_afinn_actor_plot <- average_afinn_by_actor %>%
  ggplot(aes(avg_afinn, reorder(actor, avg_afinn)))+
  geom_col(fill = "#B11313")+
  labs(x = "Average AFINN Score", y = NULL,
       title = "Average AFINN Score by Actor")+
  theme_light()
average_afinn_actor_plot
ggsave(file="average_afinn_actor_plot.svg", plot=average_afinn_actor_plot, width=7, height=4.5, path="/Users/tamnguyen/KENYON/STAT 226/project3")


## b. BING wordcloud----
library(reshape2)
library(wordcloud)
spiderman_sentiments_bing <- spiderman_sentiments %>%
  inner_join(get_sentiments("bing"), by = c(term = "word")) %>%
  count(term, sentiment, sort = T) %>%
  acast(term ~ sentiment, value.var = "n", fill = 0) %>%
  comparison.cloud(colors = c("#DF1F2D", "#2B3784"), 
                   size = 0.5,
                   title.colors = "#606c38",
                   title.size = 2.5,
                   title.bg.colors = "#FFFFFF",
                   max.words = 200)

# 2. tf-idf ----
spiderman_words <- spiderman %>%
  mutate(document = 1:nrow(.)) %>%
  unnest_tokens(word, Review) %>%
  filter(str_detect(word, "[a-z']$"),
         !word %in% stop_words$word)

spiderman_words %>%
  count(word, sort = TRUE)

spiderman_words_by_movie <- spiderman_words %>%
  count(Movie, word, sort = TRUE) %>%
  ungroup()

## a. Find tf-idf within movies----
tf_idf_by_movie <- spiderman_words_by_movie %>%
  bind_tf_idf(word, Movie, n) %>%
  arrange(desc(tf_idf))

### visualize ----
spiderman_colors <- c('#041562', '#041562', '#041562', '#9B0000', 
                      '#9B0000', '#CCCCCC', '#9B0000', '#11468F', '#11468F')

tf_idf_by_movie_plot <- tf_idf_by_movie %>%
  group_by(Movie) %>%
  slice_max(tf_idf, n = 10) %>%
  ungroup() %>%
  mutate(word = reorder_within(word, tf_idf, Movie)) %>%
  ggplot(aes(tf_idf, word, fill = Movie)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ Movie, scales = "free") +
  scale_y_reordered()+
  scale_fill_manual(values = spiderman_colors)+
  labs(x = "tf-idf", y = NULL)+
  theme_light()

tf_idf_by_movie_plot
ggsave(file="tf_idf_by_movie_plot.svg", plot=tf_idf_by_movie_plot, width=7, height=4.5, path="/Users/tamnguyen/KENYON/STAT 226/project3")


#3. Topic Modelling - LDA ----
library(topicmodels)

##prepare dtm data----
spiderman_dtm_lda <- spiderman_words %>%
  group_by(Movie, word) %>% 
  count(name = "word_total") %>% 
  ungroup() %>% 
  filter(word_total > 10) %>%
  cast_dtm(Movie, word, word_total)

## Run LDA model with 3 topics ----
spiderman_lda <- LDA(spiderman_dtm_lda, k = 9, control = list(seed = 1234))
spiderman_lda

## a. per-topic-per-word probabilities (Beta) ----
spiderman_topics <- tidy(spiderman_lda, matrix="beta")
spiderman_topics 

### Visualize: top words in each topics ----
spiderman_top_terms <- spiderman_topics %>%
  group_by(topic) %>%
  slice_max(beta, n = 15) %>%
  ungroup() %>%
  arrange(topic, -beta)

#terms that are most common within each topics
lda_beta_unigram_plot <- spiderman_top_terms %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(beta, term, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  scale_y_reordered() +
  labs(y = NULL, x = "Probability of being a part of a Topic") +
  scale_fill_manual(values = spiderman_colors_9) +
  theme_light()

ggsave(file="lda_beta_unigram_plot.svg", plot=lda_beta_unigram_plot, width=10, height=6, path="/Users/tamnguyen/KENYON/STAT 226/project3")

## b. per-document-per-topic probabilities (Gamma) ----
spiderman_documents <- tidy(spiderman_lda, matrix = "gamma")

lda_gamma_unigram_plot <- spiderman_documents %>%
  mutate(document = reorder(document, topic * gamma)) %>%
  ggplot(aes(gamma, document, fill=document)) +
  geom_col(show.legend = F) +
  facet_wrap(~ topic) +
  scale_fill_manual(values=spiderman_colors_9)+
  labs(x = "Probability of a movie belonging to a topic")+
  theme_light()

ggsave(file="lda_gamma_unigram_plot.svg", plot=lda_gamma_unigram_plot, width=10, height=6, path="/Users/tamnguyen/KENYON/STAT 226/project3")


# 4. bi-grams ----
## spiderman_bigrams: unnest_token=bigram ----
spiderman_bigrams <- spiderman %>%
  mutate(document = 1:nrow(.)) %>%
  unnest_tokens(bigram, Review, token = "ngrams", n = 2) %>%
  filter(!is.na(bigram)) %>%
  separate(bigram, c("word1", "word2"), sep = " ") %>%
  filter(!word1 %in% stop_words$word) %>%
  filter(!word2 %in% stop_words$word) %>% 
  unite(bigram, word1, word2, sep = " ")

### dtm data for bigrams----
spiderman_dtm_lda_bigrams <- spiderman_bigrams %>%
  group_by(Movie, bigram) %>% 
  count(name = "bigram_total") %>% 
  ungroup() %>% 
  filter(bigram_total > 1) %>%
  cast_dtm(Movie, bigram, bigram_total)

## a. LDA with bigrams ----
### LDA for bigrams ----
spiderman_lda_bigram <- LDA(spiderman_dtm_lda_bigrams, k = 9, control = list(seed = 5678))

### a.1. per-topic-per-word (beta) for bigrams----
spiderman_topics_bigrams <- tidy(spiderman_lda_bigram, matrix = "beta")
spiderman_colors_9 <- c('#041562', '#9B0000', '#2b3780', '#c31432', '#11468F', '#df1f2d',
                        '#004aad', '#c31432', 
                        '#041562')
lda_beta_bigrams_plot <- spiderman_topics_bigrams %>%
  group_by(topic) %>%
  slice_max(beta, n = 10) %>%
  ungroup() %>%
  arrange(topic, -beta) %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(beta, term, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  scale_y_reordered() +
  labs(y = NULL, x = "Probability of being a part of a Topic") +
  scale_fill_manual(values = spiderman_colors_9) +
  theme_light()

lda_beta_bigrams_plot
ggsave(file="lda_beta_bigrams_plot.svg", plot=lda_beta_bigrams_plot, width=12, height=5, path="/Users/tamnguyen/KENYON/STAT 226/project3")


### a.2. per-document-per-topic (gamma) for bigrams----
spiderman_documents_bigrams <- tidy(spiderman_lda_bigram, matrix = "gamma")

lda_gamma_bigrams_plot <- spiderman_documents_bigrams %>%
  mutate(document = reorder(document, topic * gamma)) %>%
  ggplot(aes(gamma, document, fill=document)) +
  geom_col(show.legend = F) +
  facet_wrap(~ topic) +
  scale_fill_manual(values=spiderman_colors_9)+
  labs(x = "Probability of a movie belonging to a topic",
       y = "Movies")+
  theme_light()

ggsave(file="lda_gamma_bigrams_plot.svg", plot=lda_gamma_bigrams_plot, width=12, height=5, path="/Users/tamnguyen/KENYON/STAT 226/project3")

## b. tf-idf with bigrams graph----
tf_idf_bigrams_plot <- spiderman_bigrams %>%
  count(Movie, bigram, sort = TRUE) %>%
  bind_tf_idf(bigram, Movie, n) %>%
  arrange(desc(tf_idf)) %>%
  group_by(Movie) %>%
  slice_max(tf_idf, n = 10) %>%
  ungroup() %>%
  mutate(bigram = reorder_within(bigram, tf_idf, Movie)) %>%
  ggplot(aes(tf_idf, bigram, fill = Movie)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ Movie, scales = "free") +
  scale_y_reordered()+
  scale_fill_manual(values = spiderman_colors)+
  labs(x = "tf-idf", y = NULL)+
  theme_light()

tf_idf_bigrams_plot
ggsave(file="tf_idf_bigrams_plot.svg", plot=tf_idf_bigrams_plot, width=10, height=6, path="/Users/tamnguyen/KENYON/STAT 226/project3")

## c. network of bigrams ----
library(igraph)
library(ggraph)

spiderman_bigrams_count <- spiderman_bigrams %>% 
  separate(bigram, c("word1", "word2"), sep = " ") %>%
  filter(!word1 %in% stop_words$word) %>%
  filter(!word2 %in% stop_words$word) %>% 
  count(word1, word2, sort = TRUE)  #generate counts of bigrams
  
spiderman_bigrams_graph <- spiderman_bigram_counts %>%
  filter(n > 50) %>%
  graph_from_data_frame()

set.seed(2017)

ggraph(spiderman_bigrams_graph, layout = "fr") +
  geom_edge_link() +
  geom_node_point() +
  geom_node_point(size = 4, color = "#bf2431", alpha=0.5)+
  geom_edge_link(aes(edge_width = n, edge_alpha = n), edge_color = "#29329b")+
  geom_node_text(aes(label = name), repel=T)+
  theme_void()+
  theme(legend.position = "bottom")

ggraph(layout = "fr") +
  geom_edge_link(
    aes(edge_width = n, edge_alpha = n), 
    edge_color = "#29329b",
    arrow = arrow(length = unit(6, 'mm'))
  ) +
  geom_node_point(size = 2, color = "#bf2431") +
  geom_node_text(aes(label = name), repel = T, 
                 point.padding = unit(0.2, "lines"),
                 vjust = 1, hjust = 1)


# draft ----
spiderman %>%
  mutate(document = 1:nrow(.)) %>%
  unnest_tokens(word, Review) %>%
  anti_join(stop_words, by=c('word'='word')) %>%
  inner_join(get_sentiments('afinn'))
  group_by(document) %>%
  nest()

  spiderman %>%
    group_by(Movie) %>%
    summarize(n=n()) %>%
    arrange(desc(n))
  
  
  
  
  
  
  
  
  spiderman_bigrams_count <- spiderman_bigrams %>%
    separate(bigram, c("word1", "word2"), sep = " ") %>%
    filter(!word1 %in% stop_words$word) %>%
    filter(!word2 %in% stop_words$word) %>% 
    count(word1, word2, sort = TRUE)
  
  spiderman_bigrams_united <- spiderman_bigrams_filtered %>%
    unite(bigram, word1, word2, sep = " ")
