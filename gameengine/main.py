import random

from gameengine.Animal import Animal
from gameengine.Money import Money
from gameengine.Player import Player

G0 = Money("Nullbauer", 0)
G10 = Money("Zehner", 10)
G50 = Money("Fuffi", 50)
G100 = Money("Hunni", 100)
G200 = Money("Zweihunni", 200)
G500 = Money("Fünfhunni", 500)

def get_random_deck():
    Tierlist = {
        "Pferd":1000,
        "Kuh":800,
        "Schwein":650,
        "Esel":500,
        "Ziege":350,
        "Schaf":250,
        "Hund":160,
        "Katze":90,
        "Gans":40,
        "Gockel":10,
    }
    TierlistO = []
    for name, wert in Tierlist.items():
        TierlistO.append(Animal(name, wert))
    Deck = []
    for tierchen in TierlistO:
        for _ in range(4):
            Deck.append(tierchen)
    random.shuffle(Deck)
    return Deck

def testDeck():
    Tierlist = {
        # "Pferd":1000,
        # "Kuh":800,
        # "Schwein":650,
        "Esel":500,
        # "Ziege":350,
        # "Schaf":250,
        # "Hund":160,
        # "Katze":90,
        # "Gans":40,
        "Gockel":10,
    }
    TierlistO = []
    for name, wert in Tierlist.items():
        TierlistO.append(Animal(name, wert))
    Deck = []
    for tierchen in TierlistO:
        for _ in range(4):
            Deck.append(tierchen)
    random.shuffle(Deck)
    return Deck

Deck = get_random_deck()
TestDeck = testDeck()


def versteigerung(verkaufender_spieler, tier, mitspieler, Deck):
    print(f"{verkaufender_spieler.name} versteigert das Tier {tier.name}.")
    if tier.name == "Esel":
        print("Der Esel wirft Geld in die Runde. (Hier muss dann noch gezählt werden wie viele Esel noch im Deck sind und dann dementspechend Geld verteilt werden)")
        # esel_count = 0 
        # for tiere in Deck:
        #     if tiere.name == "Esel":
        #         print("Esel im Deck gefunden")    # Man könnte alternativ (weniger glitches bei abbruch) nach der summe der gelder bei allen gucken? # Einfach esel counten?
        #         esel_count += 1
        # if esel_count == 4:
        #     for spieler in mitspieler:
        #         spieler.geld.append(G50)
        #         print(f"{spieler.name} bekommt 50 durch den Esel.")
        # elif esel_count == 3:
        #     for spieler in mitspieler:
        #         spieler.geld.append(G100)
        #         print(f"{spieler.name} bekommt 100 durch den Esel.")
        # elif esel_count == 2:
        #     for spieler in mitspieler:
        #         spieler.geld.append(G200)
        #         print(f"{spieler.name} bekommt 200 durch den Esel.")
        # elif esel_count == 1:
        #     for spieler in mitspieler:
        #         spieler.geld.append(G500)
        #         print(f"{spieler.name} bekommt 500 durch den Esel.")
        Geldmenge = sum(i.gges for i in mitspieler)
        Inflation = Geldmenge/len(mitspieler)-40 #Das ist das ausgezahlte geld
        index = [50,100,200,400].index(Inflation)
        schein = [G50,G100,G200,G500][index]
        for spieler in mitspieler:
            spieler.geld.append(schein)
            print(f"{spieler.name} bekommt {[50,100,200,500][index]} durch den Esel.")
        
    
    eintippung = {}
    mitspieler = [spieler for spieler in mitspieler if spieler != verkaufender_spieler]  # Verkäufer darf nicht mitbieten
    gebote = {s: 0 for s in mitspieler}
    while True:
        for spieler in mitspieler:
            eintippung[spieler] = input(f"{spieler.name} gib dein Gebot ab, oder beende die Versteigerung mit 'fertig' (aktuelles Geld: {spieler.gges}): ")
            try:
                eintippung[spieler] = int(eintippung[spieler])
            except:
                if eintippung[spieler] == "fertig" or eintippung[spieler] == "Fertig" or eintippung[spieler] == "o":
                    print(f"{spieler.name} ist raus.")
                    mitspieler.remove(spieler)
            if type(eintippung[spieler]) is type(3) and eintippung[spieler] > max(gebote.values(), default=0):
                gebote[spieler] = int(eintippung[spieler])
                print(f"{spieler.name} bietet {gebote[spieler]}.")
                print(gebote.values())
                print(f"Das höchste Gebot ist jetzt {max(gebote.values())}.")
            if mitspieler == []:
                break
        if mitspieler == []:
            break
    
    # Ermittlung des Höchstbietenden
    hoechstbietender = max(gebote, key=lambda s: gebote[s])
    hoechstes_gebot = gebote[hoechstbietender]

    print(f"{hoechstbietender.name} hat das höchste Gebot von {hoechstes_gebot} und gewinnt die Versteigerung.")
    
    hoechstbietender.Zahltag(hoechstes_gebot, verkaufender_spieler, tier)
    Deck.remove(tier)


def handeln(self, tier, herausgeforderter_spieler):
    
    #Hier wird gecheckt ob es um ein oder zwei tiere geht
    for player in [self, herausgeforderter_spieler]:
        tier_count = {}
        for tiere in player.tiere:
            if tiere == tier:
                if tiere.name in tier_count:
                    tier_count[tiere.name] += 1
                else:
                    tier_count[tiere.name] = 1
        if player == self:
            self_count = tier_count.get(tier.name, 0)
        if player == herausgeforderter_spieler:
            herausgeforderter_count = tier_count.get(tier.name, 0)
    if self_count == 2 and herausgeforderter_count == 2:
        n = 2
    else:
        n = 1
    # gebote werden abgefragt
    print(f"{self.name} handelt mit {herausgeforderter_spieler.name} um {tier.name}.")
    print(f" {self.name} bietet als Herausforderer zuerst.")
    for player in [self, herausgeforderter_spieler]:
        while True:
            gebot = input(f"{player.name}, Was sind dir die Tierchen wert? (oder 'fertig' zum Beenden) ")
            if gebot == "fertig" or gebot == "Fertig" or gebot == "0":
                print(f" {player.name} bietet {len(player.gebot)} Karten für das Tier: {tier}.")
                if player == self:
                    gebot_sum1 = sum(geldschein.wert for geldschein in player.gebot)
                    print(f"Das sind insgesamt {gebot_sum1}. als Angriff")
                else:
                    gebot_sum2 = sum(geldschein.wert for geldschein in player.gebot)
                    print(f"Das sind insgesamt {gebot_sum2}. als Verteidigung")
                break
            gefunden = False
            for geldschein in player.geld:
                if geldschein.name == gebot:
                    gefunden = True
                    player.geld.remove(geldschein)
                    player.gebot.append(geldschein)
                    print(f"Du hast einen {geldschein.name} im Wert von {geldschein.wert} zu deinem Gebot hinzugefügt.")
                    break
            if not gefunden:
                        print("Du hast diesen Schein nicht.")
    # Gewinner wird ermittelt
    if gebot_sum1 > gebot_sum2 or gebot_sum1 == gebot_sum2:
        print(f"{self.name} gewinnt das Handelsspiel um {tier.name}.")
        self.add_tier(tier)
        herausgeforderter_spieler.tiere.remove(tier)
        print("removed once")
        if n == 2:
            self.add_tier(tier)
            herausgeforderter_spieler.tiere.remove(tier)
            print("removed twice")
        for geldschein in self.gebot:
            herausgeforderter_spieler.geld.append(geldschein)
        for geldschein in herausgeforderter_spieler.gebot:
            self.geld.append(geldschein)
        self.gebot.clear()
        herausgeforderter_spieler.gebot.clear()
    elif gebot_sum1 < gebot_sum2:
        print(f"{herausgeforderter_spieler.name} gewinnt das Handelsspiel um {tier.name}.")
        herausgeforderter_spieler.add_tier(tier)
        self.tiere.remove(tier)
        if n == 2:
            herausgeforderter_spieler.add_tier(tier)
            self.tiere.remove(tier)
        for geldschein in self.gebot:
            herausgeforderter_spieler.geld.append(geldschein)
        for geldschein in herausgeforderter_spieler.gebot:
            self.geld.append(geldschein)
        self.gebot.clear()
        herausgeforderter_spieler.gebot.clear()


def Auswertung(SL):
    print("Das Spiel ist zu Ende! Hier sind die Ergebnisse:")
    for spieler in SL:
        summe = 0
        faktor = len(spieler.quattete)
        for quartett in spieler.quattete:
            #print(f"{spieler.name} hat {faktor} Quartett von {quartett} im Wert von {quartett.wert} gesammelt!")
            summe += quartett.wert
        print(f"Insgesamt hat {spieler.name} einen Wert von {summe*faktor} gesammelt.")


def Spielstart(SL, Deck):
    for i in range(len(SL)): # Startgeld kann in den Spielablauf
        for _ in range(4):                  
            SL[i].geld.append(G10)
        SL[i].geld.append(G50)
    i = 0
    while len(Deck)>0 or any(len(spieler.tiere) > 0 for spieler in SL):  # Spiel läuft weiter, solange Tiere im Deck sind oder Spieler unvollständige tiere haben
        amDransten = SL[i % len(SL)]
        print(amDransten.name + " ist am Zug. Es verbleiben " + str(len(Deck)) + " Tiere im Deck.")
        aktion = input("Was möchtest du tun? (handeln (h), versteigern (v)) ") 
        if aktion == "h" and len(amDransten.tiere) > 0 and any(len(spieler.tiere) > 0 for spieler in SL if spieler != amDransten):
            print("Deine Tiere: ")
            for idx, tier in enumerate(amDransten.tiere):
                print(f"{idx}. {tier}")
            print()
            print("Die gegnerischen Tiere: ")
            for j in range(len(SL)):
                if SL[j] != amDransten:
                    print(f"{SL[j].name}:")
                    for idx, tier in enumerate(SL[j].tiere):
                        print(f"{idx}. {tier}")
            tier_index = int(input("Wähle ein Tier zum Handeln (Nummer eingeben): "))
            if tier_index is not int:
                if 0 <= tier_index < len(amDransten.tiere):
                    tier_zu_handeln = amDransten.tiere[tier_index]
                    print("Mit wem möchtest du handeln?")
                    for idx, spieler in enumerate(SL):
                        if spieler != amDransten:
                            print(f"{idx}. {spieler.name}")
                    spieler_index = int(input("Wähle einen Spieler (Nummer eingeben): "))
                    if 0 <= spieler_index < len(SL) and SL[spieler_index] != amDransten and amDransten.tiere[tier_index] in SL[spieler_index].tiere:
                        herausgeforderter_spieler = SL[spieler_index] 
                        print(f"Du handelst mit {herausgeforderter_spieler.name}.")
                        handeln(amDransten, tier_zu_handeln, herausgeforderter_spieler)
                        print("Handel erfolgreich durchgeführt")
                        print()
                        for spieler in SL:
                            spieler.sort_tiere()
                        i += 1
                    else:
                        print("Du kannst dich nicht selbst herausfordern. / Dein Gegner muss das Tier auch besitzen.")
            else:
                print("Bitte eine gültige Nummer eingeben.")
        elif aktion == "h":
            print("Ungültige Aktion. (Ein Handel ist nur möglich, wenn du Tiere hast und ein Mitspieler auch.)")
        if aktion == "v" and len(Deck) > 0:
            versteigerung(amDransten, Deck[0], SL, Deck)
            print("Versteigerung erfolgreich durchgeführt")
            print()
            for spieler in SL:
                spieler.sort_tiere()
            i += 1
        elif aktion == "v":
            print("Ungültige Aktion. (Versteigern ist nur möglich, wenn noch Tiere im Deck sind.)") # Löst auch aus wenn ungültig oben
    Auswertung(SL)


Spieleranzahl = 3
SL = [Player('Spieler ' + str(i)) for i in range(Spieleranzahl)]

def main() -> None:
    # Spielstart(SL, Deck)
    Spielstart(SL, TestDeck)

if __name__ == "__main__":
    main()