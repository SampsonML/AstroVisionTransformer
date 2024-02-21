## Vision transformers for missing source detection
### Matt Sampson, very loose ideas

The idea is to train a ViT on a set of residual maps which respresent the difference between
astronomical observation and deblended renders. The training set will consist of mock 
observations where we intentionally miss identifying sources to leave structure in the residual map.
The ViT will be trained to classify if a missing source exists, and where so it should be put.
