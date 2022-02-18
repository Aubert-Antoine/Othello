import tkinter as tk
import numpy as np
from numba import jit

###############################################################################
#
# création de la fenetre principale
TAILLE_GRILLE = 8
ETAT_PARTIE = 1

LARG = 100 * TAILLE_GRILLE
HAUT = 100 * TAILLE_GRILLE

debug = True
deepIni = 6

ColorG  = "White"
ColorIA = "Black"
ColorH  = "White"

Window = tk.Tk()
Window.geometry(str(LARG)+"x"+str(HAUT))   # taille de la fenetre
Window.title("Othello")


TabWall = []
TAilleWall = TAILLE_GRILLE -1
for x in range(TAILLE_GRILLE) : 
        TabWall.append((x,TAilleWall))   # ici les coins sont compte 2* 
        TabWall.append((TAilleWall,x))   
        TabWall.append((TAilleWall - x,0))
        TabWall.append((0,TAilleWall - x))
TabWall = np.array(TabWall)

# création de la frame principale stockant toutes les pages
F = tk.Frame(Window)
F.pack(side="top", fill="both", expand=True)
F.grid_rowconfigure(0, weight=1)
F.grid_columnconfigure(0, weight=1)

# gestion des différentes pages
ListePages  = {}
PageActive = 0

def CreerUnePage(id):
    Frame = tk.Frame(F)
    ListePages[id] = Frame
    Frame.grid(row=0, column=0, sticky="nsew")
    return Frame

def AfficherPage(id):
    global PageActive
    PageActive = id
    ListePages[id].tkraise()
    
Frame0 = CreerUnePage(0)

"""
translates an rgb tuple of int to a tkinter friendly color code
"""
def From_rgb(rgb):
 
    r, g, b = rgb
    return f'#{r:02x}{g:02x}{b:02x}'

canvas = tk.Canvas(Frame0,width = LARG, height = HAUT, bg = From_rgb((0,86,27)))
canvas.place(x=0,y=0)

###############################################################################
#
# gestion du joueur humain et de l'IA

def PlayH(x,y):
    global Grille ,ETAT_PARTIE
    if np.sum(Grille==2)==0 :
        ETAT_PARTIE = 3 
        return
    coupsPos = CoupsPossible(Grille, 1)
    nbCoups = coupsPos[0]
    if nbCoups == 0 :
        ETAT_PARTIE = 2
        return
    for C in range (nbCoups) :
        if x==coupsPos[2*C+1] and y == coupsPos[2*C+2] :
            break
    else : return

    if(Grille[x][y] != 0) : return           
    Grille[x][y] = 1

    Grille = RetournePion(Grille, x,y,1)
    ETAT_PARTIE = 2
    
def PlayIA():
    global ETAT_PARTIE,Grille
    if np.sum(Grille==2)==0 :
        ETAT_PARTIE = 3 
        return

    Resultat = SimuleIA()
    if not Resultat[0] == 0 :
        Grille[Resultat[0][0]][Resultat[0][1]] = 2
        Grille = RetournePion(Grille, Resultat[0][0],Resultat[0][1],2)
    elif CoupsPossible(Grille, 1)[0] == 0 :
        ETAT_PARTIE = 3 
        return

    ETAT_PARTIE = 1
    
"""
SimuleIA() et SimuleHumain marchent de paire -- appel recursif:
Elles simulent tour a tour les coups du player ou de l'IA

arguments: 
    - deep est la profondeur de recherche
    - alpha et beta permetent de d'elaguer des recherche sur des branches dont la valorisation sera moin bonne que sur une autre branche deja explorer
    
return : 
    - [0,valorisation(Grille)]
    - type(valorisation(Grille)) = int
"""
def SimuleIA(deep = deepIni, alpha = float('-inf'), beta = float('inf')):
    global Grille
    coupsPos = CoupsPossible(Grille, 2)
    nbCoups = coupsPos[0]
    if estFinPartie(Grille, True) or nbCoups == 0 or deep == 0:
        return [0,valorisation(Grille)]
    
    resultats = []

    for C in range (nbCoups) :
        GrilleBis = Grille.copy()
        Grille[coupsPos[2*C+1]] [coupsPos[2*C+2]] = 2
        Grille = RetournePion(Grille, coupsPos[2*C+1],coupsPos[2*C+2],2)
        R = SimuleHumain(deep - 1, alpha, beta )[1]
        resultats.append( [(coupsPos[2*C+1],coupsPos[2*C+2]),R] )
        Grille = GrilleBis
        alpha = max(alpha, R )
        if(beta <= alpha):
            break
    
    res = [resultats[i][1] for i in range (len(resultats))]
    ind = res.index(max(res))
    return resultats[ind]

def SimuleHumain(deep, alpha, beta) :
    global Grille
    coupsPos = CoupsPossible(Grille, 1)
    nbCoups = coupsPos[0]

    if estFinPartie(Grille, True) or nbCoups == 0 or deep == 0:
        return [0,valorisation(Grille)]
    
    resultats = []

    for C in range(nbCoups) :
        GrilleBis = Grille.copy()
        Grille[coupsPos[2*C+1]] [coupsPos[2*C+2]] = 1
        Grille = RetournePion(Grille, coupsPos[2*C+1],coupsPos[2*C+2],1)
        R = SimuleIA(deep -1, alpha, beta)[1]
        resultats.append( [(coupsPos[2*C+1],coupsPos[2*C+2]),R] )
        Grille = GrilleBis
        beta = min(beta, R)
        if(beta <= alpha):
            break
    
    res = [resultats[i][1] for i in range (len(resultats))]
    ind = res.index(min(res))
    return resultats[ind]

# RetournePion appel RetournePionComp dans toutes les directions possible
@jit
def RetournePion(gCourante, x,y,player):
    GrilleRetour = gCourante.copy()

    GrilleRetour = RetournePionComp( min(TAILLE_GRILLE-x,TAILLE_GRILLE-y), GrilleRetour, x, y,  1, 1,player )
    GrilleRetour = RetournePionComp(TAILLE_GRILLE-x,                       GrilleRetour, x, y,  1, 0,player )
    GrilleRetour = RetournePionComp( min( TAILLE_GRILLE-x,y+1),            GrilleRetour, x, y,  1,-1,player )
    GrilleRetour = RetournePionComp(TAILLE_GRILLE-y,                       GrilleRetour, x, y,  0, 1,player )
    GrilleRetour = RetournePionComp(y+1,                                   GrilleRetour, x, y,  0,-1,player )
    GrilleRetour = RetournePionComp(min(x+1,TAILLE_GRILLE-y),              GrilleRetour, x, y, -1, 1,player )
    GrilleRetour = RetournePionComp(x+1,                                   GrilleRetour, x, y, -1, 0,player )
    GrilleRetour = RetournePionComp(min(x,y)+1,                            GrilleRetour, x, y, -1,-1,player )
               
    return GrilleRetour

#RetournePionComp retourne les pions necessaires dans une direction(dX, dY)
@jit
def RetournePionComp(taille, GrilleRetour, X, Y , dx, dy, player):
    
    GrilleTmp = GrilleRetour.copy()
    for i in range (1, taille ):
        if GrilleRetour[X+i*dx][Y+i*dy] == player :
            return GrilleTmp
        if GrilleRetour[X+i*dx][Y+i*dy] == 0 :
            break
        GrilleTmp[X+i*dx][Y+i*dy] = player

    return GrilleRetour

@jit
def CoupsPossible(GCourante,player):
    coups =np.array([0]*64)
    nbBlanc = np.sum(GCourante == 1)

    for abs in range (TAILLE_GRILLE):
        for ord in range (TAILLE_GRILLE):
            if(GCourante[abs][ord] == 0): 
                G = RetournePion(GCourante, abs,ord,player)
                if nbBlanc != np.sum(G == 1) :      # regarde si il y a un nb de blanc diff => retournement 
                    coups[coups[0]+1] = abs
                    coups[coups[0]+2] = ord
                    coups[0] += 2

    coups[0] = coups[0]//2

    return coups

"""
1. scoring
    1.1 le debut de partie  : maximiser son nombre d'emplacement possible pour etre plus performent sur la phase suivante qui est clef
    1.2 le milieu de partie : maximiser ses pions sur les bords et surtout les angles car se sont des positions plus strategique
    1.3 la fin de partie    : maximiser les points, ici on peut fait une atribution classique des points car on peut simuler jusqu'a la fin
"""
@jit 
def valorisation(GCourante,player = 2 ):
    EtatFinDebut = 12+deepIni
    EtatFinInter = TAILLE_GRILLE**2 - deepIni
    val = 0

    sumGrille = np.sum(GCourante != 0)
    if  (sumGrille < EtatFinDebut) :
        return len(CoupsPossible(GCourante, player))       # maximise nos positions possible 
    elif(sumGrille < EtatFinInter) :
        for i, y in TabWall:
            if(GCourante[i][y] == player ): val +=1        # calcul le meilleur ratio entre nombre de pion et position
        return np.sum(GCourante == player) + val
    else :
        return np.sum(GCourante == player)                 # calcul la meilleur fin en maximisant les pions de couleur IA

  
"""
test si nous somme dans une configuration de fin de Partie
le param 2 "simulation" est vrai si estFinPartie() est appele par simuleX()
"""
def estFinPartie(Grille, simulation = False):
    global ETAT_PARTIE
    sumGrille0 = np.sum(Grille == 0)
    if sumGrille0 == 0 :
        if( not simulation ): ETAT_PARTIE = 3          
        return True

def FinPartie():
    global Grille, ETAT_PARTIE, ColorG, ColorIA, ColorH
    ETAT_PARTIE = 0
    ColorG = "Yellow"

    sumGrille1 = np.sum(Grille == 1)
    sumGrille2 = np.sum(Grille == 2)

    if sumGrille2 > sumGrille1      : ColorIA = "Blue"
    elif sumGrille2 < sumGrille1    : ColorH ="Blue"
    else : 
        ColorIA = "white"
        ColorH ="black"

    print ("L'IA à {} points et le joueur à {} points".format(str(sumGrille2), str(sumGrille1)))

def NouvellePartie(): # Initialise la grille de jeu avec la taille donner dans TAILLE_GRILLE 
    global ColorG, ColorH, ColorIA, Grille
    ColorG ="White"
    ColorH ="White"
    ColorIA="Black"

    #grille reglementaire d'Othello
    if TAILLE_GRILLE == 8 : Grille = [  [0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0],
                                        [0,0,0,1,2,0,0,0],
                                        [0,0,0,2,1,0,0,0],
                                        [0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0]  ]

    #grille de test
    if TAILLE_GRILLE == 6 : Grille =     [  [0,0,0,0,0,0],
                                            [0,0,0,0,0,0], 
                                            [0,0,1,2,0,0], 
                                            [0,0,2,1,0,0],
                                            [0,0,0,0,0,0],
                                            [0,0,0,0,0,0] ]


    Grille = Grille = np.array(Grille)
    Grille = Grille.transpose()



################################################################################
#    
# Dessine la grille de jeu

def Dessine(PartieGagnee = False):
    ## DOC canvas : http://tkinter.fdex.eu/doc/caw.html
    canvas.delete("all")
    
    for i in range(TAILLE_GRILLE + 2):
        canvas.create_line(i*100,0,i*100,100*TAILLE_GRILLE,fill="White", width="4" )
        canvas.create_line(0,i*100,100*TAILLE_GRILLE,i*100,fill="White", width="4" )
        
    for x in range(TAILLE_GRILLE):
        for y in range(TAILLE_GRILLE):
            xc = x * 100 
            yc = y * 100 
            if ( Grille[x][y] == 1):
                canvas.create_oval(xc+10,yc+10,xc+90,yc+90,fill="white", outline=ColorH, width="4" )
            if ( Grille[x][y] == 2):
                canvas.create_oval(xc+10,yc+10,xc+90,yc+90,fill="black", outline=ColorIA, width="4" )
  
####################################################################################
#
#  fnt appelée par un clic souris sur la zone de dessin
 
def MouseClick(event):
    global ETAT_PARTIE

    if ETAT_PARTIE == 1 :
        Window.focus_set()
        x = event.x // 100  # convertit une coordonée pixel écran en coord grille de jeu
        y = event.y // 100
        if ( (x<0) or (x>TAILLE_GRILLE) or (y<0) or (y>TAILLE_GRILLE) ) : return
        PlayH(x,y)  # gestion du joueur humain et de l'IA
        estFinPartie(Grille)
        if not debug : 
            PlayIA()
            estFinPartie(Grille)

    elif ETAT_PARTIE == 2 :
        PlayIA()
        estFinPartie(Grille)

    if ETAT_PARTIE == 3 :
        FinPartie()
        ETAT_PARTIE = 0
    
    elif ETAT_PARTIE == 0 :
        NouvellePartie()
        ETAT_PARTIE = 1

    Dessine()
    
canvas.bind('<ButtonPress-1>',    MouseClick)

#####################################################################################
#
#  Mise en place de l'interface - ne pas toucher

AfficherPage(0)
NouvellePartie()
Dessine()
Window.mainloop()
