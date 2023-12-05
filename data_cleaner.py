import enchant
import os
import string



def process_words():
    d = enchant.Dict("en_US")
    directory = 'test_files'

    #need to go through the files in the directory
    for filename in os.listdir(directory):
        file_index = 1
        f = os.path.join(directory, filename)

        #hypothetically now we have a file to process

        with open(f) as file:
            text = file.read()
            words = text.split()
            print(words)
            clean_text = ""

            i = 0
            while i < len(words):
                print(words[i])
                no_punc = words[i].translate(str.maketrans('','',string.punctuation))
                if (d.check(no_punc) == False):

                    if(i < (len(words)-1)):
                        no_punc_next = words[i+1].translate(str.maketrans('','',string.punctuation))
                        print("TRYING" + " " + no_punc+no_punc_next)
                        if (d.check((no_punc+no_punc_next))):
                            print("YUH")
                            clean_text += words[i] + words[i+1] + " "
                            i+=1

                        else:
                            #otherwise try it with the previous words 
                            no_punc_prev = words[i-1].translate(str.maketrans('','',string.punctuation))
                            print("TRYING" + " " + no_punc_prev+no_punc)
                            if (d.check((no_punc_prev+no_punc))):
                                print("YUH")
                                del_ind = len(clean_text)-len(words[i-1])
                                clean_text = clean_text[0:del_ind-1]
                                clean_text += words[i-1] + words[i] + " "

                else:
                    clean_text += words[i] + " "
                
                i+=1
            # print("WORDS")
            # for word in text:
            #     print(word)


            #write cleaned data to new file
            filename = "file" + str(file_index) + ".txt"
            to_file = open("cleaned_txt_files/"+filename, "w")

            to_file.write(clean_text)
            to_file.close()
    
    file_index+=1

def main():
    process_words()



main()