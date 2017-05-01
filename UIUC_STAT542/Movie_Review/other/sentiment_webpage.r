#setwd("~/mydoc/Course/datamining/Rcode/")

myreview=read.delim("sample_movie_review.txt",
        header = T,  stringsAsFactors = F)

# remove HTML tags
myreview[,3] = gsub("<.*?>", " ", myreview[,3])

wordlist.neg=c("odd", "bad", "stupid", "shortcoming")
wordlist.pos=c("cool", "like", "talented", "great", "appreciate")

myfile = "sentiment_output.html"
if (file.exists(myfile)) file.remove(myfile)
n.review = dim(myreview)[1]

## create html file
write(paste("<html> \n", 
            "<head> \n",  
            "<style> \n",
            "@import \"textstyle.css\"", 
            "</style>", 
            "</head> \n <body>\n"), file=myfile, append=TRUE)
write("<ul>", file=myfile, append=TRUE)

for(i in 1:n.review){
  write(paste("<li><strong>", myreview[i,1], 
              "</strong> sentiment =", myreview[i,2], 
              "<br><br>", sep=" "),
        file=myfile, append=TRUE)
  tmp = strsplit(myreview[i,3], " ")[[1]]
  tmp.copy = tmp
  nwords = length(tmp)
  
  pos=NULL;
  for(j in 1:length(wordlist.neg))
    pos = c(pos, grep(wordlist.neg[j], tmp, ignore.case = TRUE))
  if (length(pos)>0) {
    for(j in 1:length(pos)){
      tmp.copy[pos[j]] = paste("<span class=\"neg\">", 
                                   tmp.copy[pos[j]], "</span>", sep="")
    }
  }
  
  pos=NULL;
  for(j in 1:length(wordlist.pos))
    pos = c(pos, grep(wordlist.pos[j], tmp, ignore.case = TRUE))
  if (length(pos)>0) {
    for(j in 1:length(pos)){
      tmp.copy[pos[j]] = paste("<span class=\"pos\">", 
                               tmp.copy[pos[j]], "</span>", sep="")
    }
  }
  
  write( paste(tmp.copy, collapse = " "), file=myfile, append=TRUE)
  write("<br><br>", file=myfile, append=TRUE)
}

write("</ul> \n  </body> \n </html>", file=myfile, append=TRUE)


