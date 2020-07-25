import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import sys

# Input Data File
if(len(sys.argv)<2):
    print("Input data file path, ex)CC-2019-CORPUS-PAPER.tsv")
    sys.exit(0)
else:
    corpus = sys.argv[1]


df = pd.read_csv(corpus, delimiter='\t')

#print(df)
print("Table 3 Raw Data")
print("Gout Flare Status as determined by Chief Complaint Prediction")
predict = df['Predict'].value_counts()
print(predict)
print("Gout Flare Status as determined by Chart Review")
consensus = df['Consensus'].value_counts()
print(consensus)


# Gout Body Locations
#Skipping stemming locations, vocabulary is small
goutBodyLocation = ['ARM','HIP','SHOULDER','WRIST','ANKLE','KNEE','TOE','FOOT','FEET','LEG','ELBOW','FINGER','THUMB']
goutBodyLocationAbbreviations = ['LLE','LUE','RUE','RLE','LE','UE']

# Past Medical History
pmhx = ['PMH','PMHX','HX','PMX']

# Gout Keywords from Table 1. https://onlinelibrary.wiley.com/doi/full/10.1002/acr.22324
gout_keywords=['gout','podagra','tophaceous','tophi','tophus']

#Alternative Gout Related Keywords from Table 1. https://onlinelibrary.wiley.com/doi/full/10.1002/acr.22324
alt_keywords=['acute flare','acute inflammatory process','allopurinol','arthritis','attack','big toe','cellulitis',
'codeine','colchicine','chronic arthritis','corticosteroids', 'diclofenac','edema','elevated levels of uric acid',
'flare','flare up','flare‐up','g6pd    ','gonagra','high uric acid level','hydrocodone','hyperuricemia','ibuprofen'
,'indomethacin','inflammation of joint','joint pain','kidney stone','king\'s disease','metacarpal',
'metacarpophalangeal joint','metatarsal phalangeal','metatarsal‐phalangeal','naprosyn','naproxen','nsaid',
'oxycodone','recu    rrent attacks','red joint','redness and swelling','swelling','swollen joint','synovial biopsy',
'synovial fluid analysis','tender joint','urate lower     drugs','urate‐lowering therapy','urate nephropathy',
'uric acid','uric acid crystals','uric crystals','voltarol','zyloric'
]

goutRegex = re.compile('.*gout.*',re.IGNORECASE)

def fetchGoutDictionaryWord(cc):
    if((re.search("|".join(gout_keywords), cc.lower())) !=None):
        findgoutkeyword = (re.search("|".join(gout_keywords), cc.lower()))
        if(findgoutkeyword!=None):
            #print(findgoutkeyword)
            return findgoutkeyword.group(0);
        return None

def fetchAltDictionaryWord(cc):
    if((re.search("|".join(alt_keywords), cc.lower())) !=None):
        findgoutkeyword = (re.search("|".join(alt_keywords), cc.lower()))
        if(findgoutkeyword!=None):
            #print(findgoutkeyword)
            return findgoutkeyword.group(0);
        return None

def hasGoutKeyword(cc):
    if((re.search("|".join(gout_keywords), cc.lower())) !=None):
        findgoutkeyword = (re.search("|".join(gout_keywords), cc.lower()))
        if(findgoutkeyword!=None):
            #print(findgoutkeyword)
            return True;
        return False

def hasAltGoutKeyword(cc):
    if((re.search("|".join(alt_keywords), cc.lower())) !=None):
        findgoutkeyword = (re.search("|".join(alt_keywords), cc.lower()))
        if(findgoutkeyword!=None):
            #print(findgoutkeyword)
            return True;
        return False

def hasGoutPmhx(cc):
    if(goutRegex.match(cc)!=None):
        findgout = (re.search('GOUT', cc.upper())).start()
        if((re.search("|".join(pmhx), cc.upper())) !=None):
            findpmhx = (re.search("|".join(pmhx), cc.upper())).start()
            if(findgout>findpmhx):
                return True
    return False

def hasGoutCurrent(cc):
    if(goutRegex.match(cc)!=None):
        findgout = (re.search('GOUT', cc.upper())).start()
        if((re.search("|".join(pmhx), cc.upper())) !=None):
            findpmhx = (re.search("|".join(pmhx), cc.upper())).start()
            if(findgout<findpmhx):
                return True
    return False


def hasGoutBodyLocationCurrent(cc):
    goutblmatch = re.search("|".join(goutBodyLocation), cc.upper())
    if(goutblmatch!=None):
        findgoutbl = goutblmatch.start()
        if((re.search("|".join(pmhx), cc.upper())) !=None):
            findpmhx = (re.search("|".join(pmhx), cc.upper())).start()
            if(findgoutbl<findpmhx):
                return True
    return False



#################################################
# Test Code
assert(hasGoutCurrent('gout flare - pmh DM'))
assert(hasGoutCurrent('bar fight, multiple abrasions - pmhx gout, HT')==False)
assert(hasGoutPmhx('bar fight, multiple abrasions - pmhx gout, HT'))
assert(hasGoutBodyLocationCurrent('knee pain - pmh DM gout'))
#################################################
#Classifiers

# Replication of Stu's results
def regexGoutClassifier(cc):
    if(goutRegex.match(cc)!=None):
        return '__label__Y'
    else:
        return '__label__N'

def regexGoutCurrentClassifier(cc):
    if(hasGoutCurrent(cc)):
        return '__label__Y'
    return '__label__N'

def regexGoutKeywordClassifier(cc):
    if(hasGoutKeyword(cc)):
        return '__label__Y'
    return '__label__N'

def regexAltGoutKeywordClassifier(cc):
    if(hasAltGoutKeyword(cc)):
        return '__label__Y'
    return '__label__N'

def regexGoutBodyLocationOrCurrentGoutClassifier(cc):
    if(goutRegex.match(cc)!=None):
        if(hasGoutBodyLocationCurrent(cc)):
            return '__label__Y'
    if(hasGoutCurrent(cc)):
        return '__label__Y'
    return '__label__N'

def regexGoutBodyLocationAndAnyGoutClassifier(cc):
    if(goutRegex.match(cc)!=None):
        if(hasGoutBodyLocationCurrent(cc)):
            return '__label__Y'
    return '__label__N'


df.rename(columns={'Chief Complaint': 'CC'}, inplace=True)


pred = df['Predict'] == 'Y'
con = df['Consensus'] == 'N'
pd.options.display.max_colwidth = 110
disagree = pd.DataFrame(df[pred & con])['CC']
#print(disagree)
#len(disagree)


def body2counts(thelist,df,thecase):
    bodydf = pd.DataFrame(df['CC'].copy())
    ccseries = pd.Series(bodydf["CC"])
    counts={}
    for item in thelist:
        bodydf[item] = ccseries.str.contains(item,regex=True,case=thecase)
    for item in thelist:
        countdf = bodydf.loc[bodydf[item] == True]
        counts[item] = len(countdf)
    return counts





train, test = train_test_split(df, test_size=0.2)
pred_labels=['__label__Y','__label__N','__label__U']


#Format data
data = df[['Consensus', 'CC']].rename(columns={"Consensus":"label", "CC":"text"})
pd.options.display.max_colwidth = 60
data['label'] = '__label__' + data['label'].astype(str)
#print(data[1:10])

data.iloc[0:int(len(data)*0.8)].to_csv('train.tsv', sep='\t', index = False, header = False)
data.iloc[int(len(data)*0.8):int(len(data)*0.9)].to_csv('test.tsv', sep='\t', index = False, header = False)
data.iloc[int(len(data)*0.9):].to_csv('dev.tsv', sep='\t', index = False, header = False);



# Test Regular Expression Classifiers
print("\nTable 5 Navie-GF and Simple-GF Performance")
def showConfusionMatrix(heading,y_true,y_pred):
    regex1data  = {'y_Actual': y_true,'y_Predicted': y_pred}
    df = pd.DataFrame(regex1data, columns=['y_Actual','y_Predicted'])
    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'],)
    print(classification_report(y_true, y_pred, target_names=pred_labels, labels=pred_labels))


y_true = data['label']
#Current Gout Classifier
y_pred = (pd.DataFrame(data['text']).applymap(regexGoutClassifier))['text']
title2display='Gout (no PMHx) Classifier (Naive GF)'
print(title2display)
showConfusionMatrix(title2display,y_true,y_pred)

#Current Gout Only Classifier
#y_pred = (pd.DataFrame(data['text']).applymap(regexGoutCurrentClassifier))['text']
#title2display='Current Gout Classifier'
#showConfusionMatrix(title2display,y_true,y_pred)

#Current Gout Or Current Body Location Classifier
y_pred = (pd.DataFrame(data['text']).applymap(regexGoutBodyLocationOrCurrentGoutClassifier))['text']
title2display='Current Gout OR (Gout Body Location And Any Gout) Classifier (SIMPLE-GF)'
print(title2display)
showConfusionMatrix(title2display,y_true,y_pred)

#Gout Or Current Body Location Classifier
#y_pred = (pd.DataFrame(data['text']).applymap(regexGoutBodyLocationAndAnyGoutClassifier))['text']
#title2display='Gout Anywhere AND Body Location Classifier'
#showConfusionMatrix(title2display,y_true,y_pred)

