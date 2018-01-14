import datetime as dt
import pickle
import random
import sys

import pygame
from sklearn import datasets
from sklearn import svm


class Pixel(pygame.sprite.Sprite):
    def __init__(self, x, y, gam):
        pygame.sprite.Sprite.__init__(self)
        self.color = (gam*10, gam*10, gam*10)
        self.image = pygame.Surface((25, 25))
        self.image.fill(self.color)
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)


class TextLabel(pygame.sprite.Sprite):
    def __init__(self, text='Test!', location=(250, 250), color=(255, 255, 255)):
        pygame.sprite.Sprite.__init__(self)
        xFont = pygame.font.Font(pygame.font.get_default_font(), 16)
        self.image = xFont.render(text, True, color)
        self.rect = self.image.get_rect()
        self.rect.center = location


class InputFrame:
    def __init__(self, oScreen, color = (105,105,105)):
        #IDEA Keep track of the image's own smaller rect paramaters, as a rect of where it is located on the main screen.
        self.drawareasize = (90, 90)
        self.image = pygame.Surface(self.drawareasize)
        self.color = color
        self.rect = self.image.get_rect()
        self.blitposition = self.rect
        self.blitposition.center = oScreen.get_rect().center
        #self.blitposition = (oScreen.get_rect().midleft[0], oScreen.get_rect().midleft[1] - (self.rect.height/2))
        self.refresh()


    def refresh(self):
        self.image.fill(self.color)
        pygame.draw.line(self.image, (128, 0, 32), (0,0), (self.drawareasize[0], 0))
        pygame.draw.line(self.image, (128, 0, 32), (0, self.drawareasize[1]/2), (self.drawareasize[0], self.drawareasize[1]/2))
        pygame.draw.line(self.image, (128, 0, 32), (0, self.drawareasize[1]), (self.drawareasize[0], self.drawareasize[1]))


class WrittenLettersDataset:
    def __init__(self):
        self.Data = {}
        self.Target = []
        self.clf = None
        self.PredictedChar = ""
        self.SamplesPerSecondWhileLearning = 0.
        self.PredictionLabel = None
        # 'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
        self.letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
               '0','1','2','3','4','5','6','7','8','9']
        self.buttons = []
        print getattr(pygame, "K_%s" % 'a')
        for cchar in self.letters:
            self.buttons.append(getattr(pygame, "K_%s" % cchar))

        ##
        #Use this to prompt for new input
        #self.letters = ['a','b','c'] #defghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789]
        self.InputFrame = None
        self.EmptyDrawSurface = None
        self.LoadData()

    def LoadData(self):
        count = 0
        try:
            print "Loading Dataset from file...."
            with open ('letters.dataset', 'rb') as data:
                self.Data = pickle.load(data)
                s = sys.getsizeof(data)
            for c in self.Data.keys():
                print "%s: %i Samples." % (c, len(self.Data[c]))
                count += len(self.Data[c])
            print "Total: %i" % count
            print "Loading Complete. S(%i)" % s

        except:
            print "No Dataset found. Blank Dataset initiated."
            self.Data = {}

    def SaveData(self):
        print "Saving Dataset..."
        with open('letters.dataset', 'wb') as output:
            pickle.dump(self.Data, output)
            s = sys.getsizeof(output)
        print "Saving Complete."
        print "Size: %i" % s

    def ClearData(self):
        print "Switching to empty dataset."
        self.Data = {}

    def AcceptNewLetterData(self, char, iframe):
        self.PredictedChar = ""
        self.clf = None
        o = self.GetScreenArray(iframe)

        if char in self.Data:
            self.Data[char].append(o)
        else:
            self.Data[char] = []
            self.Data[char].append(o)
        o = None
        s = sum(len(v) for v in self.Data.itervalues())
        if s % 10 == 0:
            self.SaveData()

    def Learn(self):
        print "Learning..."

        learningmaterial = []
        classification = []
        for ClassifiedInput in self.Data.keys():
            for pixelLists in self.Data[ClassifiedInput]:
                learningmaterial.append(pixelLists)
                classification.append(ClassifiedInput)

        #TODO: Test if i can put the below statement outside hte learning function, or if it needs to be there to learn.
        #When training an SVM with the Radial Basis Function (RBF) kernel, two parameters must be considered: C and gamma.
        #--C, common to all SVM kernels, trades off misclassification of training examples against simplicity
        #of the decision surface. A low C makes the decision surface smooth, while a high C aims at classifying all training
        #examples correctly.
        #-Gamma defines how much influance a single training session has.
        # http://scikit-learn.org/stable/modules/svm.html#svm-kernels

        #self.clf = svm.SVC(gamma=0.001, C=100.)
        self.clf = svm.SVC(gamma=0.0001)
        self.clf.fit(learningmaterial, classification)

    def GetScreenArray(self, iframe):
        o = []
        for x in xrange(0, iframe.rect.w):
            for y in xrange(0, iframe.rect.h):
                pix = iframe.image.get_at((x, y))
                if pix == (128, 0, 32, 255):  # The lines
                    d = 1
                elif pix == (255, 255, 255, 255):  # The Pen
                    d = 2
                else:  # The background
                    d = 0
                o.append(d)
                d = None
        return o

    def MakePrediction(self, iframe):
        # TODO: teach it what a empty thing looks like
        start = dt.datetime.now()
        self.Learn()
        print "Predicting..."
        input = []
        input.append(self.GetScreenArray(iframe))

        # With 43 samples, this ran in 0.292 seconds
        # With 225 samples, this ran in 4.842 seconds
        output = self.clf.predict(input)
        text = "Is this %s?" % output[0]  # output
        self.PredictedChar = output[0]
        print text
        self.PredictionLabel = TextLabel(text, (iframe.rect.size[1] / 2, 15))
        #oScreen.blit(prediction.image, prediction.rect.center)
        print "Time: %f"% (dt.datetime.now()-start).total_seconds()



class SingleLetters():
    def __init__(self, classification):
        self.classification = classification
        self.data = []


def LoadNewMatrix(digits, PixelMatrix):
    PixelMatrix.empty()
    randi = random.randint(0, len(digits.data))

    x, y = 50, 50
    # Fill a 8x8 and display to screen
    c = 0
    for n in digits.data[randi]:
        p = Pixel(x, y, n)
        PixelMatrix.add(p)
        x += 25
        c += 1
        if c > 7:
            c = 0
            y += 25
            x = 50

    return randi




def LearnFromHuman():
    #this function needs to:
    # direct the human on what to draw (a-zA-Z 0-9) each one 10 times
    # the data needs to be perm. stored somehow.
    #  -find the grid position of only the color pixels
    #   -Put them in a dictionary, the grid points as the data points.
    # After sufficent learning, allow the human to use mouse to input any charecter.
    # Predict charecter.
    # allow user to accept/correct learning function
    # store newly learned data perm.
    pygame.init()
    pygame.font.init()
    clock = pygame.time.Clock()

    # Pygame stuff
    size = 250, 250
    oScreen = pygame.display.set_mode(size)
    oScreen.fill((0, 0, 0))
    PixelMatrix = pygame.sprite.Group()

    #Load the dataset. Determine the number of class in the dataset, the classes labels, and the sample size in each class.
    #prompt user for more input as needed.
    HandWrittenLetters = WrittenLettersDataset()
    iFrame = InputFrame(oScreen)
    #TODO: Learn what an empty surface looks like and ignore it on predict
    #for x in xrange(0, iFrame.rect.w):
    #    for y in xrange(0, iFrame.rect.h):
    #        c = iFrame.image.get_at((x, y))
    #        d = c[0] * c[0] + c[1] * c[1] + c[2] * c[2] + c[3] * c[3]
    #        HandWrittenLetters.EmptyDrawSurface = d
    InkMode = False

    while True:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.MOUSEBUTTONDOWN:
                InkMode = True
                if HandWrittenLetters.clf:
                    HandWrittenLetters.AcceptNewLetterData(HandWrittenLetters.PredictedChar, iFrame)
                    iFrame.refresh()
                    oScreen.fill((0, 0, 0))  # Clear the thing.
            if event.type == pygame.MOUSEBUTTONUP:
                InkMode = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F1:
                    HandWrittenLetters.SaveData()
                elif event.key == pygame.K_F2:
                    for c in HandWrittenLetters.Data.keys():
                        print "%s: %i Samples." % (c, len(HandWrittenLetters.Data[c]))
                elif event.key == pygame.K_F3:
                    HandWrittenLetters.ClearData()
                elif event.key == pygame.K_END:
                    iFrame.refresh()
                    oScreen.fill((0, 0, 0))  # Clear the thing.
                    HandWrittenLetters.PredictedChar = ""
                    HandWrittenLetters.clf = None
                    InkMode = False

                elif event.key == pygame.K_SPACE:
                    InkMode = False
                    if HandWrittenLetters.clf is None:
                        HandWrittenLetters.MakePrediction(iFrame)
                        oScreen.blit(HandWrittenLetters.PredictionLabel.image, oScreen.get_rect().topleft)
                    else:
                        HandWrittenLetters.AcceptNewLetterData(HandWrittenLetters.PredictedChar, iFrame)
                        iFrame.refresh()
                        oScreen.fill((0, 0, 0))  # Clear the thing.

                elif event.key in HandWrittenLetters.buttons:#This accepts the letter on the keyboard and creates a pixel array to save as a sample.
                    InkMode = False
                    HandWrittenLetters.AcceptNewLetterData(chr(event.key), iFrame)
                    iFrame.refresh()
                    oScreen.fill((0, 0, 0))  # Clear the thing.


            if event.type == pygame.MOUSEMOTION:
                if pygame.mouse.get_focused():
                    if  InkMode:
                        offset = ((iFrame.rect.width/1)*-1, (iFrame.rect.height/1)*-1)
                        mousepos = tuple(map(sum, zip(pygame.mouse.get_pos(), offset)))
                        pygame.draw.circle(iFrame.image, (255, 255, 255), mousepos, 5, 0)


        oScreen.blit(iFrame.image, iFrame.blitposition)
        pygame.display.update()

def main():
    #iris = datasets.load_iris()
    digits = datasets.load_digits()

    pygame.init()
    pygame.font.init()
    clock = pygame.time.Clock()

    clf = svm.SVC(gamma=0.001, C=100.)
    clf.fit(digits.data[0:-1], digits.target[0:-1]) # This provides classified data, and the classifications.
    print digits.data[0:-1]

    #Pygame stuff
    size = 250, 250
    oScreen = pygame.display.set_mode(size)
    oScreen.fill((0, 0, 0))
    PixelMatrix = pygame.sprite.Group()

    #PreRender
    MatrixIndex = LoadNewMatrix(digits, PixelMatrix)
    input = digits.data[MatrixIndex:MatrixIndex+1] #This is the input
    text = "Is this the number %i?" % clf.predict(input)[0] #This compared the input to the classified data, and makes a prediction
    prediction = TextLabel(text, (size[1]/2, 15))
    PixelMatrix.add(prediction)
    PixelMatrix.draw(oScreen)
    pygame.display.update()

    while 1:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

            if event.type == pygame.MOUSEBUTTONUP:
                oScreen.fill((0, 0, 0))
                MatrixIndex = LoadNewMatrix(digits, PixelMatrix)
                input = digits.data[MatrixIndex:MatrixIndex + 1]  # This is the input
                text = "Is this the number %i?" % clf.predict(input)[0] #output
                prediction = TextLabel(text, (size[1]/2, 15))
                PixelMatrix.add(prediction)
                PixelMatrix.draw(oScreen)
                pygame.display.update()

                #print clf.predict(digits.data[MatrixIndex:MatrixIndex+1])
            #On Mouse button up, try to guess what the number is.

#I want to write a program that lets you draw letters with the mouse, and tries to recognize what letter you drew
#classification

if __name__ == "__main__":
    LearnFromHuman()

    pygame.quit()
    sys.exit()