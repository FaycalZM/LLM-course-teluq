---
title: "1.2 Limites des architectures séquentielles : RNN et LSTM"
weight: 3
---

## Le règne de la récurrence : Traiter le langage comme un flux

Maintenant que nous avons appris à transformer les mots en vecteurs d'adresses dans notre section précédente, une question brûlante se pose : comment faire pour que la machine comprenne une phrase entière ? Pour nous, humains, lire est un processus séquentiel. Nous lisons de gauche à droite, et chaque mot que nous rencontrons modifie notre compréhension globale de l'histoire.

Pendant des années, la réponse technologique à ce processus a été le **Réseau de Neurones Récurrent (RNN)**. L'idée est élégante : le modèle possède une "mémoire interne" (appelée état caché ou *hidden state*). À chaque étape, il prend un mot (un embedding) et le mélange avec sa mémoire de ce qu'il a lu précédemment. 🔑 **Notez bien cette intuition :** le RNN essaie de condenser tout le passé dans un seul petit vecteur qui évolue à chaque nouveau mot.

## Le problème de la disparition du gradient (Vanishing Gradient)

C'est ici que les choses se compliquent.

{{% hint warning %}}
**Attention : erreur fréquente ici !** On imagine souvent que les RNN ont une mémoire infinie. C'est faux. En pratique, à cause de la structure mathématique de la rétropropagation (l'algorithme qui permet au modèle d'apprendre), l'information s'estompe très vite.
{{% /hint %}}

**L'analogie du "Téléphone Arabe" (ou Chinese Whispers)** : Imaginez une file de 50 personnes. Vous murmurez une phrase complexe à la première. À la 50ème personne, il est fort probable que le message original soit devenu méconnaissable ou ait totalement disparu. C'est le **Vanishing Gradient** (disparition du gradient). Le modèle n'arrive plus à faire le lien entre un mot situé au début d'un long paragraphe et un mot situé à la fin. Pour un modèle de langage, cela signifie qu'il oublie le sujet de la phrase avant d'avoir atteint le verbe !

## L'architecture Encodeur-Décodeur et le goulot d'étranglement

Pour des tâches comme la traduction, nous avons utilisé des structures plus complexes. Regardez attentivement la **Figure 1-10 : Architecture RNN encoder-decoder**. Le système se divise en deux :
1.  **L'Encodeur** : Il lit la phrase source (ex: "I love llamas") et tente de transformer tout son sens en un seul et unique vecteur final : le **Context Embedding**.
2.  **Le Décodeur** : Il prend ce vecteur et essaie de "déplier" la phrase dans une autre langue.

{{< bookfig src="15.png" week="01" >}}

{{% hint danger %}}
🔑 **Je dois insister sur cette faille critique :** On appelle cela le **goulot d'étranglement (bottleneck)**. Imaginez que vous deviez résumer tout le sens d'un roman de 500 pages en une seule petite carte postale, puis qu'une autre personne doive réécrire le roman à partir de cette carte postale. C'est impossible sans perdre une quantité massive de détails. Plus la phrase est longue, plus le "Context Embedding" devient une bouillie statistique saturée.
{{% /hint %}}

## Le processus Autorégressif : Un token après l'autre

Une fois que le décodeur a reçu ce vecteur de contexte, il commence la génération. Comme vous pouvez le voir sur la **Figure 1-11 : Processus autoregressive**, la génération n'est pas instantanée. Le modèle prédit le premier mot, puis utilise ce premier mot comme entrée pour prédire le second, et ainsi de suite.

<a id="fig-1-11"></a>
{{< bookfig src="16.png" week="01" >}}

C'est ce qu'on appelle la nature **autorégressive** des modèles de langage. 🔑 **C'est un concept non-négociable :** presque tous les LLM actuels, y compris les plus puissants, fonctionnent encore sur ce principe de "boucle" où la sortie de l'étape *t* devient l'entrée de l'étape *t+1*. Le problème des RNN est que cette boucle est strictement séquentielle, ce qui rend l'entraînement désespérément lent car on ne peut pas traiter tous les mots en même temps.

## L'évolution vers les LSTM (Long Short-Term Memory)

Pour tenter de sauver les RNN, les chercheurs ont inventé les **LSTM**. Imaginez que dans chaque neurone, nous ajoutions des "portes" (gates) :
*   Une porte d'oubli (*forget gate*) pour décider ce qui n'est plus utile.
*   Une porte d'entrée (*input gate*) pour décider quelle nouvelle information stocker.
*   Une porte de sortie (*output gate*) pour filtrer ce qu'on transmet.

Bien que les LSTM aient permis de traiter des séquences plus longues (voir l'exemple de traduction "I love llamas" → "Ik hou van lama's" dans la **Figure 1-12**), ils n'ont pas résolu le goulot d'étranglement fondamental. Ils ont simplement rendu la carte postale un peu plus lisible, mais elle reste une carte postale limitée.

{{< bookfig src="17.png" week="01" >}}

## Implémentation : Un RNN simple en PyTorch

Pour bien saisir la lourdeur de cette approche, jetons un œil à la structure d'un RNN. Notez bien comment chaque état caché dépend de l'état précédent.

```python
import torch
import torch.nn as nn

# Structure d'un RNN pour traiter des séquences
# Testé pour Colab T4
class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SimpleRNN, self).__init__()
        # 1. Couche d'embeddings (vue en 1.1)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 2. La cellule RNN (La "mémoire" séquentielle)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        
        # 3. La tête de classification ou de prédiction
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, text):
        # text: [batch_size, seq_length]
        embedded = self.embedding(text)
        
        # Le RNN renvoie l'output de chaque étape et le dernier état caché
        output, hidden = self.rnn(embedded)
        
        # On utilise le dernier état caché (le fameux "context embedding")
        # pour prédire le mot suivant ou la classe
        return self.fc(hidden.squeeze(0))
```
<!-- TODO: add colab link -->

## Synthèse des faiblesses

Si je devais résumer pourquoi nous avons dû abandonner les RNN au profit de ce que vous utilisez aujourd'hui, je retiendrai trois points critiques :
1.  **L'oubli des débuts** : Même avec les LSTM, le modèle finit par perdre le contexte lointain (Disparition du gradient).
2.  **L'impossibilité de paralléliser** : Comme le mot 3 a besoin de l'état du mot 2, qui a besoin du mot 1, on ne peut pas utiliser la pleine puissance des cartes graphiques (GPU) modernes pour l'entraînement. C'est un processus linéaire dans un monde de calcul parallèle.
3.  **Le Goulot de Contexte** : Essayer de compresser toute une phrase dans un seul vecteur est une stratégie perdante pour la complexité du langage humain.

{{% hint info %}}
C'est dans cette impasse technologique qu'est née une idée folle : et si on arrêtait de forcer la machine à lire de gauche à droite ? Et si on lui donnait un mécanisme pour "regarder" n'importe quelle partie de la phrase instantanément ? C'est le saut quantique vers l'Attention que nous allons découvrir.
{{% /hint %}}
