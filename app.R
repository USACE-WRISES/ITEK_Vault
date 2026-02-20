library(shiny)
library(httr)
library(jsonlite)

API_BASE <- "http://127.0.0.1:8001"  # change if you use a different port

ui <- fluidPage(
  titlePanel("ITEK Vault"),
  sidebarLayout(
    sidebarPanel(
      textAreaInput("q", "Query", rows = 5, placeholder = "Ask a question about the ITEK Vault corpus..."),
      actionButton("go", "Run query"),
      tags$hr(),
      helpText("This Shiny app sends your query to the local ITEK Vault API."),
      helpText("Make sure the Python server is running first.")
    ),
    mainPanel(
      h4("Answer"),
      verbatimTextOutput("answer"),
      tags$hr(),
      h4("References"),
      tableOutput("refs"),
      tags$hr(),
      h4("Raw response"),
      verbatimTextOutput("raw")
    )
  )
)

server <- function(input, output, session) {
  
  resp <- eventReactive(input$go, {
    req <- list(query = input$q)
    
    r <- POST(
      url = paste0(API_BASE, "/query"),
      body = req,
      encode = "json",
      timeout(120)
    )
    
    # If the API errors, show the message
    if (http_error(r)) {
      txt <- content(r, as = "text", encoding = "UTF-8")
      stop(paste("API error:", status_code(r), txt))
    }
    
    content(r, as = "parsed", type = "application/json")
  })
  
  output$answer <- renderText({
    req(resp())
    resp()$answer
  })
  
  output$refs <- renderTable({
    req(resp())
    refs <- resp()$references
    
    # In case references comes back as NULL/empty
    if (is.null(refs) || length(refs) == 0) return(NULL)
    
    # If it's already a data.frame-like list, this will work
    as.data.frame(refs, stringsAsFactors = FALSE)
  })
  
  output$raw <- renderText({
    req(resp())
    toJSON(resp(), pretty = TRUE, auto_unbox = TRUE)
  })
}

shinyApp(ui, server)