#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

library(shiny)

importPackage <- function() {
    library(twitteR)
    library(stringr)
    library(sentimentr)
    library(dplyr)
    library(caret)
    library(tm)
    library(naivebayes)
    library(wordcloud)
    library(plotrix)
}

# Import package
importPackage()

authenticateAPI <- function() {
    # Key auth Twitter API
    consumer.api_key <- "fill your api key"
    consumer.api_secret_key <- "fill your api secret key"
    access.token <- "fill your access token"
    access.token_secret <- "fill your token secret"
    
    # Start authentication with OAuth
    setup_twitter_oauth(consumer.api_key, consumer.api_secret_key, access.token, access.token_secret)
}

# Authenticate Twitter API
authenticateAPI()

# Define UI for application that draws a histogram
ui <- fluidPage(

    # Application title
    titlePanel("Sentiment Analysis"),

    # Sidebar with a slider input for number of bins 
    sidebarLayout(
        sidebarPanel(
            textInput("searchTweet", "Enter a topic or hashtag (#) to search", "health"),
            sliderInput("maxTweet",
                        "Number of tweet that will be analysed",
                        min = 100,
                        max = 500,
                        value = 100),
            submitButton(text="Analyse")
        ),

        # Show a plot of the generated distribution
        mainPanel(
            tabsetPanel(
                tabPanel(
                    "Word count",
                    HTML(
                        "<div><h3>Most used word on this topic</h3></div>"
                    ),
                    plotOutput("wordCount")
                ),
                tabPanel(
                    "List sentiment",
                    HTML(
                        "<div><h3>List tweet and their sentiment</h3></div>"
                    ),
                    tableOutput("tableSentiment")
                ),
                tabPanel(
                    "Percentage Positive Negative",
                    HTML(
                        "<div><h3>Percentage between positive and negative words</h3></div>"
                    ),
                    plotOutput("piePlot"),
                    textOutput("positive"),
                    textOutput("negative"),
                    tags$head(tags$style("#positive, #negative {
                                            font-size: 20px
                    }"))
                ),
                tabPanel(
                    "Accuracy analysis",
                    HTML(
                        "<div><h3>Accuracy counted with confusion matrix</h3></div>"
                    ),
                    textOutput("accuracy"),
                    tags$head(tags$style("#accuracy {
                                            font-size: 40px
                    }"))
                ),
               plotOutput("distPlot")
            )
        )
    )
)

    
# Define server logic required to draw a histogram
server <- function(input, output) {
    
    sentiment.final <- reactive({
        # Scrapping tweet
        print(input$searchTweet)
        print(input$maxTweet)
        tweet.results <- searchTwitter(input$searchTweet, n = input$maxTweet, lang = "en")
        tweet.df <- twListToDF(tweet.results)
        tweet.df <- data.frame(tweet.df['text'])
        
        # Remove character in tweet
        tweet.df$text = str_replace_all(tweet.df$text, "[\\.\\,\\;]+", " ")
        tweet.df$text = str_replace_all(tweet.df$text, "http\\w+", "")
        tweet.df$text = str_replace_all(tweet.df$text, "@\\w+", " ")
        tweet.df$text = str_replace_all(tweet.df$text, "[[:punct:]]", " ")
        tweet.df$text = str_replace_all(tweet.df$text, "[[:digit:]]", " ")
        tweet.df$text = str_replace_all(tweet.df$text, "^ ", " ")
        tweet.df$text = str_replace_all(tweet.df$text, "[<].*[>]", " ")
        
        sentiment.score <- sentiment(tweet.df$text)
        sentiment.score <- sentiment.score %>% group_by(element_id) %>% summarise(sentiment = mean(sentiment))
        
        tweet.df$polarity <- sentiment.score$sentiment
        tweet.final <- tweet.df[, c('text', 'polarity')]
        
        tweet.final <- tweet.final[tweet.final$polarity != 0, ]
        tweet.final$sentiment <- ifelse(tweet.final$polarity < 0, "Negative", "Positive")
        tweet.final$sentiment <- as.factor(tweet.final$sentiment)
        
        tweet.balanced <- upSample(x = tweet.final$text, y = tweet.final$sentiment)
        names(tweet.balanced) <- c('text', 'sentiment')
        
        tweet.final$id <- seq(1, nrow(tweet.final))
        
        # Document Term Matrix
        get.dtm <- function(text.col, id.col, input.df, weighting) {
            
            # removing emoticon
            input.df$text <- gsub("[^\x01-\x7F]", "", input.df$text)
            
            # preprocessing text
            corpus <- VCorpus(DataframeSource(input.df))
            corpus <- tm_map(corpus, removePunctuation)
            corpus <- tm_map(corpus, removeNumbers)
            corpus <- tm_map(corpus, stripWhitespace)
            corpus <- tm_map(corpus, removeWords, stopwords("english"))
            corpus <- tm_map(corpus, content_transformer(tolower))
            
            dtm <- DocumentTermMatrix(corpus, control = list(weighting = weighting))
            return(list(
                "termMatrix" = dtm,
                "corpus" = corpus
            ))
        }
        
        colnames(tweet.final)[4] <- "doc_id"
        dtm <- get.dtm('text', 'id', tweet.final, "weightTfIdf")
        corpus <- dtm$corpus
        dtm <- dtm$termMatrix
        dtm.mat <- as.matrix(dtm)
        
        # Using Naive Bayes
        model <- naive_bayes(x = dtm.mat, y = tweet.final$sentiment, usekernel = TRUE)
        
        # predict using model
        preds <- predict(model, newdata = dtm.mat, type = "class")
        
        # calculate accuracy with Confusion Matrix
        cm <- confusionMatrix(preds, tweet.final$sentiment)
        accuracy <- cm$overall['Accuracy']
        
        print(accuracy)
        
        return(list(
            "tweet_final" = tweet.final,
            "prior" = model$prior,
            "accuracy" = accuracy,
            "word" = corpus
        ))
    })
    
    output$accuracy <- renderText({
        paste(toString(floor(sentiment.final()$accuracy * 100)), "%", sep = "")
    })
    
    output$wordCount <- renderPlot({
        wordcloud(
            sentiment.final()$word,
            random.offer = 'F',
            max.words = 50,
            col = rainbow(100),
            main="wordCount",
            scale=c(8, .5)
        )
    })
    
    # Render output
    x <- reactive({
        x <- c(sentiment.final()$prior[['Negative']], sentiment.final()$prior[['Positive']])
    })
    labels <- c("Negative", "Positive")
    output$piePlot <- renderPlot({
        pie3D(x(), labels = labels, explode = 0.1, main = "Pie chart positive and negative sentiment")
    })
    
    output$negative <- renderText(
        paste("Negative : ", 
              toString(floor(sentiment.final()$prior[['Negative']] * 100)), "%", sep = "")
    )
    
    output$positive <- renderText(
        paste("Positive : ", 
              toString(floor(sentiment.final()$prior[['Positive']] * 100)),  "%", sep = "")
    )
    
    sentimentPerTweet <- reactive({
        sentimentPerTweet <- sentiment.final()$tweet_final %>% select('text', 'sentiment')
    })
    
    output$tableSentiment <- renderTable(
        sentimentPerTweet()
    )
}

# Run the application 
shinyApp(ui = ui, server = server)
