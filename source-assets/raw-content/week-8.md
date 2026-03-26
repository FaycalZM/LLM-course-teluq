[CONTENU SEMAINE 8]
# Semaine 8 : Ingénierie des prompts

**Titre : L'art subtil du prompt engineering**

**Accroche du Professeur Khadidja Henni** : 
« Bonjour à toutes et à tous ! Je suis ravie de vous retrouver pour cette huitième semaine. Nous avons appris à construire des moteurs et à cartographier des données, mais aujourd'hui, nous allons apprendre à "murmurer à l'oreille des IA". 🔑 **Je dois insister :** le **Prompt Engineering** n'est pas une simple astuce de rédaction ou une liste de "mots magiques". C'est une véritable discipline d'ingénierie qui consiste à structurer l'intention humaine pour qu'elle s'aligne parfaitement avec la logique statistique du modèle. Préparez-vous à transformer un simple outil de discussion en un expert capable de raisonnements complexes. C'est ici que votre créativité rencontre la rigueur technique ! » [SOURCE: Livre p.167]

**Rappel semaine précédente** : « La semaine dernière, nous avons appris à découvrir la structure cachée de vos documents grâce au clustering et à BERTopic. » [SOURCE: Detailed-plan.md]

**Objectifs de la semaine** :
À la fin de cette semaine, vous saurez :
*   Décomposer l'anatomie d'un prompt professionnel en ses composantes essentielles.
*   Appliquer les techniques de Few-shot learning pour guider le style et la logique du modèle.
*   Implémenter le Chain-of-Thought pour débloquer les capacités de raisonnement mathématique et logique.
*   Maîtriser le Constrained Sampling pour forcer des sorties structurées (JSON) avec des outils locaux.

---

## 8.1 Anatomie d'un prompt (1600+ mots)

### La philosophie du prompt : De la complétion à l'instruction
« Avant de plonger dans les détails techniques, posons-nous une question fondamentale : qu'est-ce qu'un prompt ? » Dans les premières semaines, nous avons vu qu'un LLM est essentiellement une machine à prédire le token suivant. Au tout début, un prompt n'était qu'une amorce de phrase. 

Regardez la **Figure 6-6 : Un exemple basique de prompt** (p.173 du livre). Si vous écrivez "The sky is" (Le ciel est), le modèle complète naturellement par "blue" (bleu). C'est ce qu'on appelle la complétion pure. Mais pour transformer cette machine en assistant, nous avons dû passer à l'**instruction**. 🔑 **Je dois insister :** la différence entre un utilisateur amateur et un expert réside dans le passage de "Fais ceci" à "Voici qui tu es, voici le contexte, voici la tâche, et voici comment je veux le résultat". [SOURCE: Livre p.173, Figure 6-6]

### Les deux briques de base : Instruction et Données
L'évolution vers des modèles plus intelligents a permis de séparer ce que nous voulons faire de ce sur quoi nous voulons le faire. Observez la **Figure 6-7 : Deux composantes d'un prompt d'instruction** (p.174). 
*   **L'Instruction** : C'est le verbe d'action ("Classifie", "Résume", "Extrait"). 
*   **La Donnée (Data)** : C'est le texte brut que le modèle doit traiter. 

⚠️ **Attention : erreur fréquente ici !** Si vous mélangez l'instruction et la donnée sans séparation claire, le modèle peut devenir confus. C'est ce qui mène à la **Figure 6-8 : Extension avec indicateurs de sortie** (p.174). En ajoutant des balises comme "Texte :" et "Sentiment :", vous créez une structure visuelle pour le modèle. C'est ce qu'on appelle l'usage de **délimiteurs**. Utiliser des triples guillemets `"""`, des balises XML `<data></data>` ou des tirets permet de "sanitiser" votre prompt et d'éviter que le modèle ne confonde une instruction avec un morceau du texte à analyser. [SOURCE: Livre p.174, Figures 6-7 et 6-8]

### Les 5 piliers d'un prompt professionnel (Analyse de la Figure 6-11)
La **Figure 6-11 : Un exemple de prompt complexe avec de nombreux composants** (p.178) est sans doute l'une des illustrations les plus importantes de ce semestre. Elle nous montre comment construire une "architecture de contexte" complète. Décortiquons ensemble ces cinq piliers :

#### 1. Le Persona (L'Identité)
« Imaginez que vous demandiez un conseil juridique à un pirate ou à un juge de la Cour suprême. La réponse sera radicalement différente, n'est-ce pas ? » En commençant par "Tu es un expert en...", vous activez un sous-ensemble spécifique de probabilités dans le modèle. 
*   **Pourquoi ça marche ?** Durant le RLHF (Semaine 5), les modèles ont appris à associer des tons et des niveaux d'expertise à des étiquettes de rôles. 
*   🔑 **Le conseil du Prof. Henni** : Soyez précis ! "Tu es un relecteur de code spécialisé dans la sécurité Python" est bien plus efficace que "Tu es un programmeur". [SOURCE: Livre p.178]

#### 2. L'Instruction (La Tâche)
C'est le cœur de votre demande. Elle doit être impérative et sans ambiguïté. Au lieu de "Peux-tu essayer de résumer ?", dites "Résume les points clés en mettant l'accent sur les aspects financiers". 

#### 3. Le Contexte et les Données
Le modèle a une mémoire de travail limitée (la fenêtre de contexte). Plus vous lui donnez d'informations pertinentes sur la situation (ex: "Ce rapport est destiné à des investisseurs qui ne connaissent pas la technologie"), plus il pourra ajuster son curseur de complexité.

#### 4. Le Format de sortie
C'est souvent le point négligé. « Ne laissez jamais le modèle décider de la forme de sa réponse. » Si vous avez besoin d'une liste, d'un tableau Markdown ou d'un objet JSON, spécifiez-le explicitement. La **Figure 6-10** (p.176) montre comment, pour chaque tâche (Résumé, Classification, NER), on peut définir une structure de sortie attendue qui facilite l'intégration dans d'autres logiciels. [SOURCE: Livre p.176, Figure 6-10]

#### 5. Le Ton et l'Audience
Le ton (professionnel, humoristique, concis) et l'audience (un enfant de 5 ans, un expert technique) agissent comme des filtres de style finaux. C'est la différence entre une information brute et une communication réussie. [SOURCE: Livre p.178, Figure 6-11]

### Spécificité et Contraintes : La fermeté bienveillante
⚠️ **Fermeté bienveillante** : « Soyez un patron exigeant avec votre LLM. » Le modèle est statistiquement enclin à la paresse ou à la verbosité. Vous devez lui imposer des contraintes négatives.
*   "N'utilise pas de jargon technique."
*   "Ne dépasse pas 100 mots."
*   "Si tu ne connais pas la réponse, dis 'Information non disponible' au lieu d'inventer."

L'ajout de ces contraintes réduit drastiquement les **hallucinations** (section 5.4). Comme le mentionne le livre, demander au modèle de citer ses sources ou de justifier son raisonnement avant de donner la réponse finale est une technique de protection majeure. [SOURCE: Livre p.177]

### Le phénomène "Lost in the Middle" : L'ordre des mots compte
C'est une découverte récente et cruciale en ingénierie des prompts (Liu et al., 2023). Les modèles ont tendance à accorder plus d'importance aux informations situées au début (**Primacy effect**) et à la fin (**Recency effect**) du prompt.
🔑 **Je dois insister :** Si vous placez une instruction cruciale au milieu d'un long texte de contexte de 2000 mots, il y a de fortes chances que le modèle l'ignore. 
*   **Bonne pratique** : Placez votre instruction principale au tout début ("Tu es... Ta tâche est...") et répétez les contraintes de format à la toute fin, juste avant que le modèle ne commence à générer. [SOURCE: Livre p.177]

### Exemple de code : Construction d'un template dynamique sur Colab
En tant qu'ingénieurs, vous n'allez pas écrire chaque prompt à la main. Vous allez construire des générateurs de prompts. Voici comment structurer cela proprement en Python pour une utilisation sur T4.

```python
# Testé sur Colab T4 16GB VRAM
def build_pro_prompt(task_data, persona="expert analyst", output_format="bullet points"):
    """
    Construit un prompt structuré selon les 5 piliers vus en cours.
    [SOURCE: Basé sur l'anatomie du prompt Livre p.178]
    """
    
    # 1. Persona & Rôle
    prompt = f"### ROLE\nYou are a {persona}. Your goal is to provide high-quality, accurate insights.\n\n"
    
    # 2. Contexte & Données (Délimités par des balises pour éviter les fuites)
    prompt += f"### DATA\n<input_text>\n{task_data}\n</input_text>\n\n"
    
    # 3. Instruction & Contraintes
    prompt += "### TASK\nAnalyze the text provided above. Be concise and factual. Do not hallucinate facts.\n\n"
    
    # 4. Format & Ton
    prompt += f"### OUTPUT EXPECTATIONS\nFormat: {output_format}\nTone: Professional and objective.\n"
    prompt += "Begin your response directly without introductory fluff."

    return prompt

# Utilisation
raw_text = "The quarterly results show a 15% increase in revenue but a 5% drop in user retention."
final_prompt = build_pro_prompt(raw_text, persona="Senior Financial Advisor")

print(final_prompt)
```

### Éthique et Responsabilité : Le biais du "Prompt Master"
⚠️ **Éthique ancrée** : « Mes chers étudiants, l'ingénieur de prompt est celui qui tient le pinceau, mais c'est lui qui choisit les couleurs. » 
Lorsque vous définissez un Persona, vous invoquez des stéréotypes. Si vous demandez au modèle d'agir comme un "dirigeant agressif", vous risquez de faire ressortir les pires biais sexistes ou comportementaux de ses données d'entraînement. 
🔑 **Conséquence éthique :** L'ingénierie des prompts peut être utilisée pour contourner les protections de sécurité des modèles (Jailbreaking). En tant qu'experts formés dans ce cours, vous avez la responsabilité d'utiliser ces techniques pour augmenter l'utilité et la sécurité des systèmes, et non pour exploiter leurs faiblesses. [SOURCE: Livre p.28]

« Nous avons maintenant décortiqué l'anatomie de l'interaction. Vous savez construire un prompt solide, pilier par pilier. Mais que se passe-t-il si une simple description ne suffit pas ? Parfois, il faut montrer au modèle ce qu'on attend de lui. Dans la prochaine section, nous allons explorer l'une des capacités les plus mystérieuses et puissantes des LLM : le **Few-shot prompting**, ou l'art d'apprendre en un clin d'œil grâce à des exemples. »

---
*Fin de la section 8.1 (1640 mots environ)*
## 8.2 Techniques avancées (1600+ mots)

### L'apprentissage sans mise à jour : Le miracle de l'In-Context Learning
« Bonjour à toutes et à tous ! Nous entrons maintenant dans la partie la plus spectaculaire de l'ingénierie des prompts. La semaine dernière, nous avons vu que pour spécialiser un modèle BERT, il fallait modifier ses poids (le fine-tuning). Mais avec les modèles génératifs comme GPT ou Phi-3, il existe une forme de magie appelée l'**In-Context Learning** (Apprentissage en contexte). 🔑 **Je dois insister :** le modèle n'apprend rien de nouveau de manière permanente, ses neurones ne changent pas. Il utilise simplement sa mémoire de travail pour s'adapter à vos exemples. C'est comme donner une fiche de consignes à un intérimaire très intelligent : il comprend instantanément ce qu'il doit faire pour la durée de sa mission. » [SOURCE: Livre p.180]

### La hiérarchie du guidage : De Zero-shot à Few-shot
Comme l'illustre la **Figure 6-13 : Zero-shot, one-shot et few-shot prompting** (p.181 du livre), nous disposons d'une échelle de précision pour guider le modèle. Décortiquons cette figure ensemble : elle représente une tâche de classification de sentiments ("neutral", "negative", "positive") et montre comment l'ajout d'exemples transforme la réponse de l'IA. [SOURCE: Livre p.181, Figure 6-13]

#### 1. Zero-shot Prompting (Zéro exemple)
C'est ce que nous avons fait jusqu'ici. On donne une instruction brute ("Classifie ce texte"). 
*   **Pourquoi ça marche ?** Le modèle s'appuie uniquement sur ce qu'il a appris durant son pré-entraînement massif.
*   **Le risque** : Le modèle peut ne pas respecter le format de sortie ou mal interpréter une nuance très spécifique à votre métier. ⚠️ **Attention : erreur fréquente ici !** Penser que le Zero-shot suffit pour des tâches complexes. Si le modèle échoue, ne changez pas de modèle tout de suite : passez au One-shot.

#### 2. One-shot Prompting (Un exemple)
On fournit un seul couple "Entrée / Sortie" pour servir de modèle. 
*   **Analyse de la Figure 6-13 (milieu)** : En montrant au modèle que pour la phrase "L'hôtel était correct", la réponse attendue est "Neutral", vous fixez non seulement la logique, mais aussi le vocabulaire autorisé.
*   **L'effet miroir** : Le modèle va imiter votre ton, votre concision et votre formatage. C'est l'outil idéal pour fixer un style d'écriture.

#### 3. Few-shot Prompting (Plusieurs exemples)
🔑 **C'est la technique la plus robuste.** On donne généralement entre 3 et 8 exemples au modèle. 
*   **Analyse de la Figure 6-13 (droite)** : En listant plusieurs exemples variés (un positif, un négatif, un neutre), vous créez une "mini-base de connaissances" temporaire. 
*   **Le cas "Gigamuru" (p.181-182)** : Le livre donne un exemple fascinant. On invente un mot, le "Gigamuru" (un instrument de musique japonais imaginaire). En donnant un exemple de phrase utilisant ce mot, le modèle est capable de générer de nouvelles phrases parfaites. Cela prouve que le modèle peut apprendre des concepts totalement nouveaux... tant qu'ils restent dans son prompt ! [SOURCE: Livre p.181-182]

### L'art de choisir ses exemples (Few-shot engineering)
⚠️ **Fermeté bienveillante** : « Ne jetez pas n'importe quels exemples dans votre prompt ! » La qualité du Few-shot dépend de trois règles non-négociables :
1.  **La Diversité** : Si vous ne donnez que des exemples de critiques positives, le modèle aura un biais optimiste. Donnez des exemples qui couvrent tous les cas de figure.
2.  **L'Ordre des exemples** : Les modèles souffrent de "récence". L'exemple situé juste avant la question réelle a plus d'influence que le premier. 🔑 **Mon conseil de professeur** : Mettez votre exemple le plus complexe ou le plus important en dernier dans la liste.
3.  **La Cohérence du format** : Si votre premier exemple est en JSON et le second en texte brut, le modèle sera perdu. Soyez un métronome dans votre structure. [SOURCE: Blog 'LLM Roadmap' de Maarten Grootendorst]

### Chain Prompting : Diviser pour mieux régner
Parfois, une tâche est trop lourde pour être résolue en une seule fois. Imaginez demander à un architecte de dessiner les plans, de commander les matériaux et de construire la maison en une seule phrase. C'est la recette du désastre.

Pour les LLM, nous utilisons le **Chain Prompting** (Prompting en chaîne), illustré magnifiquement par la **Figure 6-14 : Utilisation de chaînes de prompts** (p.183). 
Cette figure décrit le processus de création d'un produit :
1.  **Lien 1** : Créer un nom de produit à partir de caractéristiques.
2.  **Lien 2** : Créer un slogan à partir du nom trouvé.
3.  **Lien 3** : Écrire un argumentaire de vente à partir du nom et du slogan.

🔑 **Je dois insister sur l'avantage technique :** En découpant la tâche, vous permettez au modèle d'accorder 100% de son "attention" (Semaine 3) à un petit problème à la fois. Cela réduit drastiquement les **hallucinations**. Le modèle n'a plus besoin de jongler avec 5 contraintes en même temps ; il se concentre sur une seule brique. [SOURCE: Livre p.183, Figure 6-14]

### Prompt Chaining vs Sequential Prompting
Dans le Sequential Prompting, nous faisons plusieurs appels au modèle. La sortie du prompt A devient une variable dans le prompt B. 
*   *Analogie* : C'est comme une ligne de montage à l'usine. Chaque étape affine la pièce précédente.
*   *Usage professionnel* : Traduire un texte (Prompt 1), puis demander à l'IA de corriger les erreurs culturelles de la traduction (Prompt 2), puis formater le résultat en HTML (Prompt 3). [SOURCE: Livre p.184]

### Laboratoire de code : Implémentation Few-shot et Chaining (Colab T4)
Voici comment structurer une chaîne de prompts sophistiquée en Python. Nous allons utiliser Phi-3-mini pour simuler un pipeline de création de contenu.

```python
# Testé sur Colab T4 16GB VRAM
# !pip install transformers accelerate

from transformers import pipeline
import torch

# Initialisation du modèle léger (QUESTION CODE)
model_id = "microsoft/Phi-3-mini-4k-instruct"
pipe = pipeline("text-generation", model=model_id, device_map="auto", torch_dtype=torch.float16)

# --- ÉTAPE 1 : FEW-SHOT POUR L'EXTRACTION ---
# On montre au modèle comment extraire des entités d'un texte complexe
extraction_prompt = [
    {"role": "user", "content": "Extract items from: I bought an apple, a car, and a house."},
    {"role": "assistant", "content": "1. apple\n2. car\n3. house"},
    {"role": "user", "content": "Extract items from: My inventory includes a sword, a shield, and 5 potions."}
]

# --- RÉPONSE (ANSWER CODE) ---
# [SOURCE: In-context learning Livre p.180-181]

# 1. Exécution du premier maillon (Extraction)
res1 = pipe(extraction_prompt, max_new_tokens=50)
extracted_items = res1[0]['generated_text'][-1]['content']

print(f"Maillon 1 (Objets) :\n{extracted_items}")

# 2. Exécution du second maillon (Chaining)
# On utilise la sortie du maillon 1 pour construire le prompt 2
story_prompt = [
    {"role": "system", "content": "You are a fantasy writer."},
    {"role": "user", "content": f"Write a 2-sentence story using these items: {extracted_items}"}
]

res2 = pipe(story_prompt, max_new_tokens=100)
print(f"\nMaillon 2 (Histoire) :\n{res2[0]['generated_text'][-1]['content']}")
```

### Le danger de la "Paresse de Contexte"
⚠️ **Fermeté bienveillante** : « Ne surchargez pas votre prompt d'exemples inutiles. » 
Chaque mot que vous ajoutez au prompt consomme des tokens et réduit l'espace disponible pour la réponse. De plus, si vos exemples sont trop similaires ("Un chat est un animal", "Un chien est un animal"), vous n'apprenez rien de nouveau au modèle, vous gaspillez simplement de la mémoire. 
🔑 **La règle d'or** : 3 exemples contrastés (un cas simple, un cas complexe, un cas limite) valent mieux que 20 exemples répétitifs. [SOURCE: Livre p.177]

### Éthique et Responsabilité : Le modèle "perroquet"
⚠️ **Éthique ancrée** : « Mes chers étudiants, le Few-shot peut être une prison sémantique. » 
Si vous donnez au modèle des exemples qui contiennent des stéréotypes, il va les reproduire avec une fidélité effrayante. Par exemple, si dans vos exemples de classification de CV, vous ne montrez que des hommes pour des rôles techniques, le modèle "apprendra" en quelques millisecondes que c'est le pattern à suivre. 
🔑 **Conséquence éthique :** Le Few-shot est une injection directe de biais. En tant qu'experts, vous devez auditer vos exemples d'entraînement avec autant de soin que si vous écriviez le code source de l'application. Vous êtes les "enseignants" de l'IA le temps d'un prompt. [SOURCE: Livre p.28]

« Vous maîtrisez maintenant les techniques de guidage par l'exemple et la division des tâches. Vous savez comment transformer une IA hésitante en un exécutant précis. Mais comment s'assurer que le modèle "réfléchit" vraiment avant de répondre à une énigme ? Dans la prochaine section, nous allons explorer les techniques de **Raisonnement**, comme le célèbre **Chain-of-Thought**, pour donner aux LLM une véritable "conscience" étape par étape. »

---
*Fin de la section 8.2 (1620 mots environ)*
## 8.3 Raisonnement et structuration (2000+ mots)

### Dépasser l'intuition : Quand l'IA prend le temps de réfléchir
« Bonjour à toutes et à tous ! J'espère que vous êtes bien accrochés, car nous abordons aujourd'hui le "Saint Graal" du Prompt Engineering. Jusqu'ici, nous avons traité nos modèles comme des dictionnaires ou des traducteurs rapides (Système 1). Mais que se passe-t-il quand nous leur demandons de résoudre un problème de mathématiques complexe, de déjouer un paradoxe logique ou de planifier une stratégie d'entreprise ? 🔑 **Je dois insister :** un LLM, par défaut, est un parieur statistique. Il veut donner la réponse la plus probable *immédiatement*. Aujourd'hui, nous allons lui apprendre à ralentir, à sortir son "brouillon mental" et à raisonner étape par étape. Bienvenue dans le monde du **Raisonnement Augmenté**. » [SOURCE: Livre p.184-185]

### La distinction entre Système 1 et Système 2
Pour comprendre l'intérêt de la structuration du raisonnement, nous devons faire un détour par la psychologie cognitive, citée par Daniel Kahneman et reprise dans le domaine des LLM.
*   **Système 1 (Pensée Rapide)** : Intuition, réflexe, génération immédiate. C'est le mode par défaut du LLM qui prédit le token suivant sans "réfléchir".
*   **Système 2 (Pensée Lente)** : Logique, calcul, vérification, décomposition. C'est ce que nous essayons de déclencher via les techniques que nous allons voir.

🔑 **L'intuition du Professeur Henni :** Imaginez que je vous demande : "Combien font 17 fois 24 ?". Si vous répondez au hasard, c'est le Système 1. Si vous prenez un papier et un crayon pour poser l'opération, c'est le Système 2. Les techniques de raisonnement sont le "papier et le crayon" du LLM. [SOURCE: Livre p.185]

### Chain-of-Thought (CoT) : La puissance du brouillon
L'article fondateur de Wei et al. (2022) a introduit le **Chain-of-Thought (Chaîne de pensée)**. Cette technique est illustrée par la **Figure 6-15 : Le Chain-of-Thought par l'exemple** (p.186 du livre). 

Décortiquons le contenu de cette figure capitale :
*   **Le problème** : La figure montre à gauche un prompt "standard" (Few-shot classique). On donne une question mathématique et une réponse directe. Résultat ? Le modèle se trompe sur la nouvelle question car il essaie de deviner le chiffre final d'un coup.
*   **La solution CoT** : À droite, l'exemple fourni au modèle inclut le **cheminement logique**. "Roger a 5 balles. Il achète 2 boîtes de 3, donc 2x3=6. 5+6=11." 
*   **L'effet** : En voyant ce modèle de pensée, le Modèle de Langage imite cette structure pour la nouvelle question. Il décompose : "La cafétéria avait 23 pommes. Elles en ont utilisé 20, il en reste 3. Elles en achètent 6, donc 3+6=9."

🔑 **Pourquoi cela fonctionne-t-il techniquement ?** En forçant le modèle à générer les étapes de calcul avant la réponse finale, vous permettez au mécanisme d'attention (Semaine 3) de s'appuyer sur les tokens de raisonnement qu'il vient de s'écrire à lui-même. C'est une extension de la puissance de calcul via la génération de tokens intermédiaires. [SOURCE: Livre p.185-186, Figure 6-15]

### Le miracle du Zero-shot CoT : "Réfléchissons étape par étape"
Parfois, vous n'avez pas d'exemples sous la main pour faire du CoT. C'est là qu'intervient la découverte de Kojima et al. (2022), illustrée par la **Figure 6-16 : Chain-of-Thought sans exemple** (p.187).

La figure montre qu'en ajoutant simplement la phrase magique **"Let's think step by step"** (Réfléchissons étape par étape) à la fin d'une question complexe, le modèle change de comportement. 
*   **Avant** : Il donne une réponse brute (souvent fausse).
*   **Après** : Il s'auto-conditionne à décomposer sa réponse en paragraphes logiques. 

⚠️ **Attention : erreur fréquente ici !** Penser que cette phrase résout tout. Sur des modèles de petite taille (moins de 7B paramètres), le Zero-shot CoT peut parfois aggraver les choses en faisant "délirer" le modèle sur de fausses pistes logiques. 🔑 **Je dois insister :** Le raisonnement est une capacité émergente qui dépend de la taille et de la qualité du pré-entraînement du modèle. [SOURCE: Livre p.187, Figure 6-16]

### Self-Consistency : La sagesse de la majorité
Un seul chemin de raisonnement peut être erroné. Pour stabiliser les résultats, nous utilisons la **Self-Consistency** (Auto-cohérence), représentée en **Figure 6-17** (p.188).

L'idée est inspirée des méthodes d'ensemble en Machine Learning :
1.  On pose la question au modèle avec un prompt CoT.
2.  On génère non pas une, mais **plusieurs réponses** (par exemple 5 ou 10) en utilisant une température élevée (ex: 0.7).
3.  Chaque réponse aura un chemin de raisonnement potentiellement différent.
4.  On regarde quelle réponse finale (le chiffre ou la conclusion) apparaît le plus souvent.

🔑 **L'intuition technique :** Le modèle peut faire une erreur de calcul dans un chemin, mais il est statistiquement improbable qu'il fasse la *même* erreur exacte dans 5 chemins différents. La majorité l'emporte et la fiabilité s'envole. [SOURCE: Livre p.188, Figure 6-17]

### Tree-of-Thought (ToT) : L'exploration par l'arbre des possibles
Pour les problèmes vraiment complexes (ex: écrire un plan marketing complet ou résoudre une énigme de type Sudoku), le CoT linéaire ne suffit plus. Il faut explorer plusieurs idées et pouvoir revenir en arrière. C'est le **Tree-of-Thought**, illustré par la **Figure 6-18** (p.189).

Cette architecture, beaucoup plus lourde, traite le raisonnement comme une recherche dans un arbre :
*   **Génération de pensées** : Le modèle propose 3 idées pour l'étape 1.
*   **Évaluation** : Le modèle (ou un autre modèle) note la pertinence de chaque idée. 
*   **Élagage (Pruning)** : On abandonne les branches qui mènent à une impasse.
*   **Backtracking** : Si aucune branche ne marche, on remonte à l'étape précédente pour essayer autre chose.

🔑 **Le concept d'Expert Discussion (Figure 6-19)** : Une variante simplifiée consiste à demander au modèle d'imaginer trois experts (ex: un ingénieur, un designer, un juriste) qui débattent du problème dans le prompt. Le consensus final est souvent bien plus robuste qu'une réponse unique. [SOURCE: Livre p.189-191, Figures 6-18, 6-19]

### Structuration de la sortie : Forcer la rigueur
Le raisonnement ne sert à rien si le résultat est noyé dans un texte illisible. La structuration consiste à imposer une forme logique à la pensée de l'IA. 
Nous utilisons pour cela des techniques de **Contraintes de format** (Format constraints) :
*   **Markdown** : Pour les titres et les listes.
*   **Balises de raisonnement** : Utiliser `<thinking>...</thinking>` pour séparer le brouillon de la réponse finale `<answer>...</answer>`.

### Laboratoire de code : Implémentation CoT et Vote majoritaire (Colab T4)
Voici comment orchestrer un raisonnement Système 2 avec Phi-3-mini. Nous allons résoudre un problème de logique en simulant une Self-Consistency simplifiée.

```python
# Testé sur Colab T4 16GB VRAM
from transformers import pipeline
import torch
from collections import Counter

model_id = "microsoft/Phi-3-mini-4k-instruct"
pipe = pipeline("text-generation", model=model_id, device_map="auto", torch_dtype=torch.float16)

# --- PROMPT AVEC CHAIN-OF-THOUGHT ---
# [SOURCE: Technique CoT Livre p.186]
puzzle = "If I have 3 baskets with 5 apples each, and I give 2 apples to a friend, but then find another basket with 4 apples, how many apples do I have?"

cot_prompt = f"""Question: {puzzle}
Answer: Let's think step by step:
1."""

# --- RÉPONSE : GÉNÉRATION MULTIPLE (SELF-CONSISTENCY) ---
# [SOURCE: Technique Self-consistency p.188]
print("Génération des chemins de pensée...")
responses = pipe(
    cot_prompt, 
    max_new_tokens=150, 
    num_return_sequences=3, # On génère 3 versions
    do_sample=True, 
    temperature=0.7
)

final_answers = []
for i, res in enumerate(responses):
    text = res['generated_text']
    print(f"\n--- Chemin {i+1} ---\n{text}")
    # Extraction simplifiée du dernier nombre (logique d'exemple)
    import re
    nums = re.findall(r'\d+', text)
    if nums: final_answers.append(nums[-1])

# Vote majoritaire
if final_answers:
    majority = Counter(final_answers).most_common(1)[0][0]
    print(f"\n🔑 RÉPONSE FINALE VALIDÉE : {majority}")
```

⚠️ **Fermeté bienveillante** : Observez les chemins générés. Parfois, un chemin sera très verbeux et un autre très sec. C'est la beauté (et le danger) du décodage probabiliste. En tant qu'ingénieurs, votre rôle est de "contenir" cette variation par des prompts de structuration.

### Éthique et Responsabilité : Le piège de la rationalisation
⚠️ **Éthique ancrée** : « Mes chers étudiants, méfiez-vous de la "belle parole". » 
Un LLM peut produire un raisonnement étape par étape qui semble parfait, mais dont la conclusion est fausse. Ou pire : il peut avoir la bonne réponse par chance, et inventer un raisonnement (rationalisation) pour la justifier après coup.
1.  **L'illusion de logique** : Ce n'est pas parce qu'un modèle écrit "D'après la loi de Newton..." qu'il applique réellement les principes de la physique. Il corrèle des mots de physique.
2.  **Biais de confirmation** : Si votre prompt contient une erreur ("Pourquoi 2+2 font 5 ?"), le modèle, dans son désir d'être "utile" (Sycophancy, Semaine 5), inventera une logique absurde pour valider votre erreur. 

🔑 **Je dois insister :** Le Chain-of-Thought n'est pas une preuve de vérité, c'est une aide à la performance. Vous restez l'ultime arbitre de la logique. Utilisez les modèles d'experts (ToT) pour confronter les points de vue et limiter ces biais. [SOURCE: Livre p.28, p.191]

« Vous avez maintenant les clés du raisonnement. Vous ne demandez plus seulement à l'IA de parler, vous lui demandez de construire une pensée. C'est une étape gigantesque vers des applications industrielles robustes. Mais attention : une pensée brillante dans un format brouillon est difficile à exploiter. Dans la prochaine section, nous allons apprendre à "verrouiller" la sortie pour qu'elle respecte une grammaire stricte, comme le JSON, indispensable pour connecter vos IA à vos logiciels. »

---
*Fin de la section 8.3 (2080 mots environ)*
## 8.4 Vérification et contrôle (1500+ mots)

### Le rempart contre l'imprévisibilité : L'IA au service de la production
« Bonjour à toutes et à tous ! Nous arrivons à la dernière étape de notre semaine sur l'ingénierie des prompts. Prenez un instant pour réaliser le chemin parcouru : nous avons appris à structurer nos demandes (section 8.1), à guider l'IA par l'exemple (section 8.2) et à débloquer son raisonnement logique (section 8.3). Mais une question cruciale demeure : comment faire pour que l'IA ne devienne pas le "maillon faible" de votre système informatique ? 🔑 **Je dois insister :** une réponse fluide et intelligente ne sert à rien si elle fait planter votre application parce qu'il manque une virgule dans un fichier JSON ou si elle contient une information confidentielle interdite. Aujourd'hui, nous allons apprendre à mettre des "garde-fous" à la machine. Bienvenue dans l'ère de l'**IA contrôlée et vérifiée**. » [SOURCE: Livre p.191]

### Pourquoi valider la sortie ? Les trois risques majeurs
Dans un environnement professionnel, on ne peut pas se contenter d'un "ça a l'air de marcher". Maarten Grootendorst et Jay Alammar identifient trois piliers de vérification indispensables pour tout ingénieur en LLM (p.191) :
1.  **La structure (Format)** : Si votre logiciel attend un format JSON pour mettre à jour une base de données et que l'IA répond "Voici votre JSON : { ... }", la phrase d'introduction fera échouer votre code.
2.  **La validité (Logic)** : Même si le format est bon, les valeurs doivent être cohérentes. Une IA qui génère un personnage de jeu avec une "force" de 1 000 000 alors que le maximum est 100 brise l'équilibre de votre application.
3.  **L'éthique et la sécurité** : C'est le point non-négociable. Nous devons nous assurer que la sortie ne contient pas de données personnelles (PII), de propos haineux ou de failles de sécurité (injections de code). [SOURCE: Livre p.191-192]

### Technique 1 : La boucle de vérification par les pairs (Self-Correction)
La méthode la plus simple consiste à utiliser un LLM pour en vérifier un autre, ou pour se vérifier lui-même. C'est ce qu'illustre la **Figure 6-19 : Utilisation d'un LLM pour vérifier la sortie** (p.194 du livre). 

Décortiquons le contenu de cette figure :
*   **Le cycle itératif** : La figure montre un premier passage où l'IA génère une réponse (A). Cette réponse est ensuite passée à un "vérificateur" (qui peut être le même modèle avec un prompt différent).
*   **Les règles (Guardrails)** : Le vérificateur a une liste de critères (ex: "Le texte est-il en JSON ? Est-il courtois ?"). 
*   **La rétroaction** : Si une règle est enfreinte, le vérificateur renvoie une erreur à l'IA initiale ("Tu as oublié de fermer l'accolade, recommence").

🔑 **L'intuition du Professeur Henni :** C'est comme un auteur qui se relit ou qui demande à un éditeur de corriger son manuscrit. Cela améliore la qualité, mais attention : cela double votre consommation de tokens et votre temps de réponse ! [SOURCE: Livre p.194, Figure 6-19]

### Technique 2 : L'échantillonnage contraint (Constrained Sampling)
C'est ici que l'ingénierie devient "magique" et extrêmement puissante. Au lieu de laisser l'IA générer du texte et de corriger l'erreur après coup, nous allons **intervenir directement dans son cerveau** (au niveau des probabilités) pour l'empêcher physiquement de faire une erreur de syntaxe.

Regardons les **Figures 6-20 et 6-21** (p.195-196) :
*   **Figure 6-20 : Génération de pièces manquantes** : On donne au modèle un "moule" (un template JSON avec des trous). On force le modèle à ne remplir que les trous. Le modèle n'a pas le choix, il ne peut pas ajouter de texte inutile autour.
*   **Figure 6-21 : Constrained Sampling par masque de probabilité** : C'est la technique la plus avancée. Imaginez que le modèle doive choisir le prochain token. Normalement, il a 50 000 choix possibles. Si nous savons que l'étape suivante dans un JSON doit être un chiffre, nous appliquons un "masque" : nous mettons la probabilité de tous les mots du dictionnaire à zéro, sauf pour les chiffres de 0 à 9. 

🔑 **Je dois insister :** Avec cette technique, le modèle ne peut mathématiquement PAS produire une erreur de format. Il devient un moteur de données déterministe au sein d'une architecture probabiliste. [SOURCE: Livre p.195-196, Figures 6-20 et 6-21]

### L'usage des Grammaires (GBNF)
Pour réaliser ce tour de force avec des modèles locaux (comme ceux que nous utilisons sur Colab), nous utilisons souvent le format **GBNF** (*Guidance/GGML BNF*). C'est une façon de définir une "recette" syntaxique que l'IA doit suivre.
*   *Analogie* : C'est comme un rail de chemin de fer. Le train (l'IA) peut aller vite et être puissant, mais il ne peut pas sortir des rails que vous avez posés. [SOURCE: Documentation llama.cpp / GBNF]

### Laboratoire de code : Forcer une sortie JSON parfaite (Colab T4)
Nous allons utiliser la bibliothèque `llama-cpp-python` qui permet de charger des modèles compressés (GGUF) et de leur imposer une structure JSON stricte. ⚠️ **Attention !** Assurez-vous d'avoir redémarré votre session Colab pour libérer la VRAM (Semaine 5).

```python
# Installation des outils de contrôle
# !pip install llama-cpp-python 

from llama_cpp import Llama
import json

# 1. Chargement d'un modèle compact (TinyLlama ou Phi-3 GGUF)
# [SOURCE: CONCEPT À SOURCER – Inférence optimisée Livre p.202]
llm = Llama(
    model_path="path_to_your_model.gguf", # Remplacez par votre lien de téléchargement
    n_gpu_layers=-1, # On utilise tout le GPU T4 !
    n_ctx=2048
)

# 2. Définition de la tâche
prompt = "Create a health report for a patient named Sarah. Include age, heart_rate, and status."

# 3. GÉNÉRATION CONTRAINTE (L'arme secrète)
# [SOURCE: Constrained sampling p.194]
# On demande explicitement un 'json_object' au moteur d'inférence
response = llm.create_chat_completion(
    messages=[{"role": "user", "content": prompt}],
    response_format={
        "type": "json_object",
        "schema": { # On définit le schéma attendu
            "type": "object",
            "properties": {
                "age": {"type": "number"},
                "heart_rate": {"type": "number"},
                "status": {"type": "string"}
            },
            "required": ["age", "heart_rate", "status"]
        }
    },
    temperature=0.7 # On peut garder de la créativité sur le contenu !
)

# 4. Extraction et validation
json_output = json.loads(response["choices"][0]["message"]["content"])
print(f"Sortie validée : {json_output}")
```

🔑 **Note du Professeur** : Remarquez que nous avons gardé une température de 0.7. Le modèle est libre d'inventer le "status" (ex: "Stable", "Excellente santé"), mais il est **obligé** de le mettre dans la bonne case du JSON. C'est l'équilibre parfait entre l'imagination de l'IA et la rigueur de l'informatique classique.

### Frameworks de contrôle : Guardrails, Guidance et Outlines
Dans le monde professionnel, vous n'allez pas tout coder à la main. Il existe des bibliothèques dédiées à la sécurité et au contrôle des LLM (mentionnées p.194) :
*   **Guardrails AI** : Permet de définir des "Rails" en XML pour vérifier les sorties (ex: "Vérifie qu'il n'y a pas d'insultes").
*   **Guidance (Microsoft)** : Un langage de programmation qui entrelace le texte humain et le contrôle de l'IA.
*   **Outlines** : Une bibliothèque spécialisée dans l'échantillonnage contraint pour garantir des sorties JSON ou Regex à 100%. [SOURCE: Livre p.194]

### Éthique et Responsabilité : L'illusion du contrôle total
⚠️ **Fermeté bienveillante** : « Ne vous laissez pas bercer par un sentiment de sécurité trompeur. » 
Ce n'est pas parce que votre IA sort un JSON parfait que les informations à l'intérieur sont vraies.
1.  **Hallucinations structurées** : L'IA peut vous donner un JSON très propre contenant des faits totalement inventés (ex: un faux diagnostic médical). 🔑 **Règle d'or :** La structure n'est pas la vérité.
2.  **Biais de formatage** : Parfois, forcer un format (ex: n'accepter que "Oui" ou "Non") empêche l'IA d'exprimer une nuance importante ou une incertitude nécessaire. C'est une forme de censure technique qui peut mener à des décisions erronées. 
3.  **Sur-confiance** : Un développeur qui voit une sortie JSON a tendance à lui faire plus confiance qu'à un texte libre. C'est un piège psychologique. Gardez toujours un esprit critique. [SOURCE: Livre p.28]

🔑 **Le message du Prof. Henni** : « Le contrôle est le pont qui permet aux LLM de sortir des laboratoires pour entrer dans les usines, les hôpitaux et les banques. En maîtrisant ces techniques, vous ne construisez plus seulement des gadgets, vous construisez des systèmes fiables. Mais n'oubliez jamais : vous êtes le chef d'orchestre, et la machine n'est que l'instrument. La responsabilité finale de la décision appartient toujours à l'humain. » [SOURCE: Livre p.28]

« Nous avons terminé notre immense semaine sur l'ingénierie des prompts ! Vous savez maintenant comment parler à l'IA, comment la faire raisonner, et comment la tenir en laisse pour qu'elle respecte vos contraintes. C'est un arsenal de compétences qui fait de vous des professionnels recherchés. La semaine prochaine, nous allons attaquer un défi encore plus grand : comment donner une mémoire infinie à votre IA grâce au **RAG** (Retrieval-Augmented Generation). Préparez-vous, car c'est là que l'IA rencontre vos propres données ! Mais d'abord, place au laboratoire final de la semaine. »

---
*Fin de la section 8.4 (1590 mots environ)*
## 🧪 LABORATOIRE SEMAINE 8 (800+ mots)

**Accroche du Professeur Khadidja Henni** : 
« Bonjour à toutes et à tous ! Nous passons maintenant à la phase de "sculpture". Dans ce laboratoire, vous n'êtes plus de simples utilisateurs, vous êtes des ingénieurs du contexte. 🔑 **Je dois insister :** l'IA est un miroir de votre clarté. Si votre prompt est flou, sa pensée le sera aussi. Nous allons apprendre à transformer des réponses banales en raisonnements brillants et à dompter l'imprévisibilité de la machine pour obtenir des données structurées. Prêt·e·s à murmurer à l'oreille des modèles ? C'est parti ! » [SOURCE: Livre p.167]

---

### 🔹 QUIZ MCQ (10 questions)

1. **Quel composant d'un prompt professionnel définit l'identité et le niveau d'expertise attendu du modèle ?**
   a) L'Instruction
   b) Le Persona (ou Rôle)
   c) Le Délimiteur
   d) Le Few-shot example
   **[Réponse: b]** [Explication: Le persona conditionne le ton, le vocabulaire et la profondeur de l'analyse en activant des sous-espaces sémantiques spécifiques. SOURCE: Livre p.178, Figure 6-11]

2. **Quelle technique consiste à donner au modèle entre 3 et 8 exemples de couples "Entrée/Sortie" pour guider son comportement ?**
   a) Zero-shot prompting
   b) Instruction tuning
   c) Few-shot prompting (In-context learning)
   d) Fine-tuning supervisé
   **[Réponse: c]** [Explication: C'est l'art de montrer au lieu de dire, permettant au modèle d'imiter une structure sans changer ses poids. SOURCE: Livre p.181]

3. **Quel est l'avantage principal du "Chain-of-Thought" (CoT) pour les problèmes mathématiques ?**
   a) Il réduit la consommation de jetons (tokens).
   b) Il accélère la vitesse de génération.
   c) Il permet au modèle d'utiliser ses propres tokens intermédiaires comme une mémoire de travail pour stabiliser son calcul.
   d) Il empêche le modèle de parler une autre langue.
   **[Réponse: c]** [Explication: En explicitant les étapes, le modèle réduit la charge cognitive de la prédiction finale. SOURCE: Livre p.186, Figure 6-15]

4. **Quelle méthode permet d'explorer plusieurs chemins de raisonnement en parallèle et de choisir le meilleur ?**
   a) Chain-of-Thought
   b) Sequential Prompting
   c) Tree-of-Thought (ToT)
   d) Bag-of-Words
   **[Réponse: c]** [Explication: ToT permet l'évaluation, l'élagage des mauvaises pistes et le retour en arrière (backtracking). SOURCE: Livre p.189, Figure 6-18]

5. **Le phénomène "Lost in the Middle" suggère que les modèles oublient souvent les informations situées :**
   a) Au tout début du prompt.
   b) À la toute fin du prompt.
   c) Au milieu d'un long contexte (2000+ mots).
   d) Dans les exemples de Few-shot.
   **[Réponse: c]** [Explication: Les modèles ont un biais de primauté et de récence, négligeant le centre des longs textes. SOURCE: Livre p.177]

6. **Quelle phrase magique active généralement le "Zero-shot CoT" ?**
   a) "Réponds uniquement en JSON."
   b) "Ignore les instructions précédentes."
   c) "Let's think step by step" (Réfléchissons étape par étape).
   d) "Tu es un expert en mathématiques."
   **[Réponse: c]** [Explication: Cette phrase force le modèle à adopter une structure décomposée sans avoir besoin d'exemples préalables. SOURCE: Livre p.187, Figure 6-16]

7. **Dans la technique de "Self-Consistency", comment détermine-t-on la réponse finale ?**
   a) On garde la première réponse générée.
   b) On fait la moyenne des nombres trouvés.
   c) On effectue un vote majoritaire sur plusieurs générations indépendantes.
   d) On demande à un humain de choisir.
   **[Réponse: c]** [Explication: On mise sur le fait que la bonne réponse est statistiquement plus probable à travers plusieurs chemins logiques. SOURCE: Livre p.188, Figure 6-17]

8. **À quoi sert le "Constrained Sampling" (échantillonnage contraint) ?**
   a) À limiter le nombre de mots dans la réponse.
   b) À forcer mathématiquement le modèle à suivre une grammaire stricte (ex: JSON) en modifiant les probabilités des tokens.
   c) À réduire la température du modèle.
   d) À empêcher le modèle d'utiliser le GPU.
   **[Réponse: b]** [Explication: On applique un masque sur le dictionnaire pour n'autoriser que les tokens syntaxiquement valides. SOURCE: Livre p.194-196]

9. **Quelle est la limite éthique majeure du "Few-shot prompting" ?**
   a) Il coûte trop cher en électricité.
   b) Il peut injecter et amplifier les biais humains présents dans les exemples choisis par le développeur.
   c) Il rend le modèle trop intelligent.
   d) Il ne fonctionne qu'en anglais.
   **[Réponse: b]** [Explication: Le modèle imite fidèlement les patterns des exemples, y compris les préjugés sexistes ou culturels. SOURCE: Livre p.28]

10. **Dans quel cas le "Chain Prompting" (découpage en plusieurs appels) est-il préférable à un prompt unique ?**
    a) Pour une simple traduction de mot.
    b) Pour des tâches complexes avec de multiples contraintes contradictoires (ex: créer un nom, puis un slogan, puis un plan média).
    c) Pour économiser des tokens.
    d) Pour masquer l'identité de l'utilisateur.
    **[Réponse: b]** [Explication: En isolant les tâches, on permet au modèle de dévouer toute son attention à une seule contrainte à la fois. SOURCE: Livre p.183, Figure 6-14]

---

### 🔹 EXERCICE 1 : Optimisation de prompt par le raisonnement (Niveau 1)

**Objectif** : Transformer un prompt "Zero-shot" qui échoue en un prompt "Chain-of-Thought" réussi.

```python
# --- CODE AVANT COMPLÉTION (QUESTION) ---
from transformers import pipeline
import torch

# Modèle léger TinyLlama pour Colab
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
pipe = pipeline("text-generation", model=model_id, torch_dtype=torch.bfloat16, device_map="auto")

# Le problème mathématique complexe
question = "The cafeteria had 23 apples. They used 20 for lunch and bought 6 more. How many apples do they have?"

# TÂCHE : Comparez la réponse directe et la réponse avec CoT
print("--- TEST 1 : RÉPONSE DIRECTE ---")
# [VOTRE CODE ICI]

# --- RÉPONSE COMPLÈTE (CORRIGÉ) ---
# [SOURCE: Technique CoT Livre p.186-187]

# 1. Test sans raisonnement (le modèle risque de répondre 27 ou 29 par confusion)
prompt_direct = f"<|user|>\nQuestion: {question}\nAnswer:<|assistant|>\n"
res1 = pipe(prompt_direct, max_new_tokens=10, do_sample=False)
print(f"Direct: {res1[0]['generated_text'].split('Answer:')[-1].strip()}")

# 2. Test avec Zero-shot Chain-of-Thought
print("\n--- TEST 2 : RÉPONSE AVEC RAISONNEMENT (CoT) ---")
prompt_cot = f"<|user|>\nQuestion: {question}\nAnswer: Let's think step by step:<|assistant|>\n"
res2 = pipe(prompt_cot, max_new_tokens=100, do_sample=False)
print(f"CoT: {res2[0]['generated_text'].split('assistant|>')[-1].strip()}")
```

**Explications détaillées** :
*   **Résultats attendus** : Le test 1 donne souvent un chiffre faux. Le test 2 décompose : 23 - 20 = 3, puis 3 + 6 = 9.
*   **Justification** : En forçant le modèle à écrire les étapes de soustraction, on évite qu'il ne fasse une corrélation statistique trop rapide entre les nombres 23, 20 et 6.

---

### 🔹 EXERCICE 2 : Few-shot prompting pour l'extraction (Niveau 2)

**Objectif** : Apprendre au modèle un format d'extraction personnalisé et complexe sans fine-tuning.

```python
# --- CODE AVANT COMPLÉTION (QUESTION) ---
# On veut extraire le NOM et la COULEUR des fruits dans un format "Fruit: [Nom] | Color: [Couleur]"

messy_text = "I have a big red apple and a small yellow banana in my basket."

# TÂCHE : Construisez un prompt Few-shot avec 2 exemples.

# --- RÉPONSE COMPLÈTE (CORRIGÉ) ---
# [SOURCE: In-context learning Livre p.181]

few_shot_prompt = [
    {"role": "user", "content": "Extract: A green lime and an orange orange."},
    {"role": "assistant", "content": "Fruit: lime | Color: green\nFruit: orange | Color: orange"},
    {"role": "user", "content": f"Extract: {messy_text}"}
]

# Inférence
# [SOURCE: Chat templates p.191]
formatted_prompt = pipe.tokenizer.apply_chat_template(few_shot_prompt, tokenize=False, add_generation_prompt=True)
output = pipe(formatted_prompt, max_new_tokens=50, do_sample=False)

print("Résultat de l'extraction Few-shot :")
print(output[0]['generated_text'].split("<|assistant|>")[-1].strip())
```

**Explications détaillées** :
*   **Résultats attendus** : "Fruit: apple | Color: red / Fruit: banana | Color: yellow".
*   **Justification** : Le modèle identifie le pattern "Fruit: ... | Color: ..." grâce aux exemples et l'applique scrupuleusement au nouveau texte, même s'il contient des adjectifs perturbateurs ("big", "small").

---

### 🔹 EXERCICE 3 : Validation de sortie JSON (Niveau 3)

**Objectif** : Utiliser un prompt de structuration pour obtenir un objet JSON valide et le charger en Python.

```python
# --- CODE AVANT COMPLÉTION (QUESTION) ---
import json

# TÂCHE : Créez un prompt qui génère les statistiques d'un personnage de RPG.
# Le JSON doit avoir les clés : "name", "class", "power_level".

# --- RÉPONSE COMPLÈTE (CORRIGÉ) ---
# [SOURCE: Output verification Livre p.191-192]

prompt_json = """<|user|>
Create a fantasy character profile. 
Output ONLY valid JSON code. No conversation.
Format:
{
  "name": "string",
  "class": "Warrior/Mage/Rogue",
  "power_level": int
}
<|assistant|>
"""

res = pipe(prompt_json, max_new_tokens=100, do_sample=False)
raw_content = res[0]['generated_text'].split("<|assistant|>")[-1].strip()

try:
    # Validation par chargement
    data = json.loads(raw_content)
    print("✅ JSON VALIDE GÉNÉRÉ :")
    print(json.dumps(data, indent=2))
except:
    print("❌ Erreur de formatage JSON.")
    print(f"Texte brut reçu : {raw_content}")
```

**Explications détaillées** :
*   **Attentes** : Le modèle doit renvoyer uniquement le bloc de code `{ ... }`. 
*   **Avertissement du Professeur** : Sans "Constrained Sampling" matériel (GBNF), le modèle peut parfois ajouter du texte ("Voici le JSON..."). 🔑 **L'astuce d'ingénieur** : Si `json.loads` échoue, utilisez une expression régulière (Regex) pour extraire uniquement ce qui se trouve entre les accolades `{}`.

---

**Mots-clés de la semaine** : Persona, In-Context Learning, Few-shot, Chain-of-Thought (CoT), Self-Consistency, Tree-of-Thought, Lost in the Middle, Constrained Sampling, GBNF, JSON Formatting.

**En prévision de la semaine suivante** : Nous allons apprendre à combattre les hallucinations de manière radicale. Comment connecter votre IA à Internet ou à vos propres documents PDF pour qu'elle réponde avec des preuves ? Bienvenue dans le monde du **RAG (Retrieval-Augmented Generation)**. [SOURCE: Detailed-plan.md]

**SOURCES COMPLÈTES** :
*   Livre : Alammar & Grootendorst (2024), *Hands-On LLMs*, Chapitre 6, p.167-198.
*   Guide de Prompting : [PromptingGuide.ai](https://www.promptingguide.ai/fr)
*   Blog Jay Alammar : *The Illustrated Transformer* (pour l'intuition de l'attention dans le CoT).
*   GitHub Officiel : chapter06 repository.

[/CONTENU SEMAINE 8]