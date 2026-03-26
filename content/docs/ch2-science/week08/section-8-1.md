---
title: "8.1 Anatomie d'un prompt"
weight: 2
---

## La philosophie du prompt : De la complétion à l'instruction
Avant de plonger dans les détails techniques, posons-nous une question fondamentale : qu'est-ce qu'un prompt ? 

Dans les premières semaines, nous avons vu qu'un LLM est essentiellement une machine à prédire le token suivant. Au tout début, un prompt n'était qu'une amorce de phrase. 

Regardez la **Figure 8-1 : Un exemple basique de prompt** .

{{< bookfig src="139.png" week="08" >}}

Si vous écrivez "The sky is" (Le ciel est), le modèle complète naturellement par "blue" (bleu). C'est ce qu'on appelle la complétion pure. Mais pour transformer cette machine en assistant, nous avons dû passer à l'**instruction**. 

> [!IMPORTANT]
🔑 **Je dois insister :** la différence entre un utilisateur amateur et un expert réside dans le passage de "Fais ceci" à "Voici qui tu es, voici le contexte, voici la tâche, et voici comment je veux le résultat".


## Les deux briques de base : Instruction et Données
L'évolution vers des modèles plus intelligents a permis de séparer ce que nous voulons faire de ce sur quoi nous voulons le faire. Observez la **Figure 8-2 : Deux composantes d'un prompt d'instruction**. 
*   **L'Instruction** : C'est le verbe d'action ("Classifie", "Résume", "Extrait"). 
*   **La Donnée (Data)** : C'est le texte brut que le modèle doit traiter. 

{{< bookfig src="140.png" week="08" >}}

> [!WARNING]
⚠️ **Attention : erreur fréquente ici !** Si vous mélangez l'instruction et la donnée sans séparation claire, le modèle peut devenir confus. C'est ce qui mène à la **Figure 8-3 : Extension avec indicateurs de sortie**. 

{{< bookfig src="141.png" week="08" >}}

> En ajoutant des balises comme "Texte :" et "Sentiment :", vous créez une structure visuelle pour le modèle. C'est ce qu'on appelle l'usage de **délimiteurs**. Utiliser des triples guillemets `"""`, des balises XML `<data></data>` ou des tirets permet de "sanitiser" votre prompt et d'éviter que le modèle ne confonde une instruction avec un morceau du texte à analyser. 


## Les 5 piliers d'un prompt professionnel
La **Figure 8-4 : Un exemple de prompt complexe avec de nombreux composants** est sans doute l'une des illustrations les plus importantes de ce cours. Elle nous montre comment construire une "architecture de contexte" complète. Décortiquons ensemble ces cinq piliers :

{{< bookfig src="144.png" week="08" >}}

### 1. Le Persona (L'Identité)
Imaginez que vous demandiez un conseil juridique à un pirate ou à un juge de la Cour suprême. La réponse sera radicalement différente, n'est-ce pas ? 

En commençant par "Tu es un expert en...", vous activez un sous-ensemble spécifique de probabilités dans le modèle. 
*   **Pourquoi ça marche ?** Durant le RLHF ([**Semaine 5**]({{< relref "section-5-2.md" >}}#RLHF)), les modèles ont appris à associer des tons et des niveaux d'expertise à des étiquettes de rôles. 

> [!TIP]
> 🔑 **Mon conseil** : Soyez précis ! "Tu es un relecteur de code spécialisé dans la sécurité Python" est bien plus efficace que "Tu es un programmeur".

### 2. L'Instruction (La Tâche)
C'est le cœur de votre demande. Elle doit être impérative et sans ambiguïté. Au lieu de "Peux-tu essayer de résumer ?", dites "Résume les points clés en mettant l'accent sur les aspects financiers". 

### 3. Le Contexte et les Données
Le modèle a une mémoire de travail limitée (la fenêtre de contexte). Plus vous lui donnez d'informations pertinentes sur la situation (ex: "Ce rapport est destiné à des investisseurs qui ne connaissent pas la technologie"), plus il pourra ajuster son curseur de complexité.

### 4. Le Format de sortie
C'est souvent le point négligé. 

> [!WARNING]
Ne laissez jamais le modèle décider de la forme de sa réponse.

Si vous avez besoin d'une liste, d'un tableau Markdown ou d'un objet JSON, spécifiez-le explicitement. La **Figure 8-5** montre comment, pour chaque tâche (Résumé, Classification, NER), on peut définir une structure de sortie attendue qui facilite l'intégration dans d'autres logiciels.

{{< bookfig src="143.png" week="08" >}}

### 5. Le Ton et l'Audience
Le ton (professionnel, humoristique, concis) et l'audience (un enfant de 5 ans, un expert technique) agissent comme des filtres de style finaux. C'est la différence entre une information brute et une communication réussie.


## Spécificité et Contraintes : La fermeté bienveillante

> [!IMPORTANT]
⚠️ Soyez un patron exigeant avec votre LLM. 

Le modèle est statistiquement enclin à la paresse ou à la verbosité. Vous devez lui imposer des contraintes négatives.
*   "N'utilise pas de jargon technique."
*   "Ne dépasse pas 100 mots."
*   "Si tu ne connais pas la réponse, dis 'Information non disponible' au lieu d'inventer."

L'ajout de ces contraintes réduit drastiquement les **hallucinations** ([**section 5.4**]({{< relref "section-5-4.md" >}}#hallucinations)).

> [!TIP]
Demander au modèle de citer ses sources ou de justifier son raisonnement avant de donner la réponse finale est une technique de protection majeure.


## Le phénomène "Lost in the Middle" : L'ordre des mots compte
C'est une découverte récente et cruciale en ingénierie des prompts (Liu et al., 2023). Les modèles ont tendance à accorder plus d'importance aux informations situées au début (**Primacy effect**) et à la fin (**Recency effect**) du prompt.

> [!IMPORTANT]
🔑 **Je dois insister :** Si vous placez une instruction cruciale au milieu d'un long texte de contexte de 2000 mots, il y a de fortes chances que le modèle l'ignore. 

> [!TIP]
✅ **Bonne pratique** : Placez votre instruction principale au tout début ("Tu es... Ta tâche est...") et répétez les contraintes de format à la toute fin, juste avant que le modèle ne commence à générer.


## Exemple de code : Construction d'un template dynamique sur Colab
En tant qu'ingénieurs, vous n'allez pas écrire chaque prompt à la main. Vous allez construire des générateurs de prompts. Voici comment structurer cela proprement en Python pour une utilisation sur T4.

```python
# Testé sur Colab T4
def build_pro_prompt(task_data, persona="expert analyst", output_format="bullet points"):
    """
    Construit un prompt structuré selon les 5 piliers vus en cours.
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

## Éthique et Responsabilité : Le biais du "Prompt Master"

> [!WARNING]
⚠️ Mes chers étudiants, l'ingénieur de prompt est celui qui tient le pinceau, mais c'est lui qui choisit les couleurs. 
> Lorsque vous définissez un Persona, vous invoquez des stéréotypes. Si vous demandez au modèle d'agir comme un "dirigeant agressif", vous risquez de faire ressortir les pires biais sexistes ou comportementaux de ses données d'entraînement. 

> [!IMPORTANT]
🔑 **Conséquence éthique :** L'ingénierie des prompts peut être utilisée pour contourner les protections de sécurité des modèles (Jailbreaking). En tant qu'experts formés dans ce cours, vous avez la responsabilité d'utiliser ces techniques pour augmenter l'utilité et la sécurité des systèmes, et non pour exploiter leurs faiblesses.

---
Nous avons maintenant décortiqué l'anatomie de l'interaction. Vous savez construire un prompt solide, pilier par pilier. Mais que se passe-t-il si une simple description ne suffit pas ? Parfois, il faut montrer au modèle ce qu'on attend de lui. Dans la prochaine section ➡️, nous allons explorer l'une des capacités les plus mystérieuses et puissantes des LLM : le **Few-shot prompting**, ou l'art d'apprendre en un clin d'œil grâce à des exemples.