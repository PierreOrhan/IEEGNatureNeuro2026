color_syntax = "#b3539eff"
feature_names = ["phoneme","wordform","lexicalSyntactic","syntacticOperations","syntacticState","MDStree_9d","MDS_CStree_9d"]
color_hierarchy = {"syntacticState":"#ff5500ff",
                   "syntacticOperations":"#ff9500ff",
                   "lexicalSyntactic":"#f2ff00ff",
                   "wordform":"#73e205ff",
                   "phoneme":"#0cb100ff",
                   "MDStree_9d":color_syntax,
                   "MDS_CStree_9d":"#505d8ffa"}
features_to_label= {"phoneme":"Phoneme",
                   "wordform":"Wordform",
                   "syntacticOperations":"Syntactic operations",
                   "syntacticState":"Syntactic state",
                   "lexicalSyntactic":"Lexical-syntactic",
                   "MDStree_9d":"Dependency-coding features",
                   "MDS_CStree_9d":"Constituent-coding features"}
