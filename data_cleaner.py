import enchant
import os
import string



def process_words():
    d = enchant.Dict("en_US")
    directory = 'txt_files'

    #need to go through the files in the directory
    file_index = 1
    for filename in os.listdir(directory):
        
        f = os.path.join(directory, filename)

        #hypothetically now we have a file to process

        with open(f) as file:
            text = file.read()
            words = text.split()
            clean_text = ""

            i = 0
            while i < len(words):
                #stripping punctuation so the word check will work, but saving it to make sure that we don't just get rid of all punctuation
                no_punc = words[i].translate(str.maketrans('','',string.punctuation))
                if(len(no_punc) > 0):
                    if (d.check(no_punc) == False):

                        #first try combining it with the next word chunk
                        if(i < (len(words)-1)):
                            no_punc_next = words[i+1].translate(str.maketrans('','',string.punctuation))
                            if(len(no_punc_next) > 0):
                                if (d.check((no_punc+no_punc_next))):
                                    clean_text += words[i] + words[i+1] + " "
                                    i+=1

                            else:
                                #otherwise try it with the previous word bit (which may have been counted as a vlaid word)
                                if(i > 0):
                                    no_punc_prev = words[i-1].translate(str.maketrans('','',string.punctuation))
                                    if(len(no_punc_prev) > 0):
                                        if (d.check((no_punc_prev+no_punc))):
                                            del_ind = len(clean_text)-len(words[i-1])
                                            clean_text = clean_text[0:del_ind-1]
                                            clean_text += words[i-1] + words[i] + " "



                                else:
                                    #just add it anyway because sometimes courts make up words
                                    clean_text += words[i] + " "



                    else:
                        clean_text += words[i] + " "

                
                i+=1
 


            #write cleaned data to new file
            filename = "file" + str(file_index) + ".txt"
            to_file = open("cleaned_txt_files/"+filename, "w")

            to_file.write(clean_text)
            to_file.close()
    
        file_index+=1

def main():
    process_words()



main()