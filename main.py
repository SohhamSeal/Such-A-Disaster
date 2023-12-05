from PreAndPostProcessing.functions import train,test  #...

if __name__ == "__main__":
  choice=int(input("Enter 1 to train, 2 to test and anything else to exit :: "))
  if choice==1:
    #call train()
    train()
  elif choice==2:
    #call test()
    test()
  else:
    print("exiting....Thank you!!")