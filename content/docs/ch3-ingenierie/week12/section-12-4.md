---
title: "12.4 Évaluation et bonnes pratiques"
weight: 5
---


## Au-delà de la perte : Mesurer l'âme de l'assistant
Bonjour à toutes et à tous ! J'espère que vous avez survécu à la rigueur mathématique du DPO dans notre section précédente. Nous arrivons maintenant à la question qui hante tout développeur d'IA : comment savoir si notre modèle est "devenu meilleur" ? 

Dans le Machine Learning classique, nous avions des scores d'exactitude (Accuracy). En NLP traditionnel, nous avions le score BLEU. 

> [!IMPORTANT]
📌 **Je dois insister :** en alignement, ces chiffres ne valent plus rien. 

Un modèle peut avoir un score de prédiction parfait et être un assistant détestable, arrogant ou dangereux. Aujourd'hui, nous allons apprendre à mesurer l'insaisissable : la préférence humaine. Respirez, nous allons transformer des jugements subjectifs en une science de l'évaluation rigoureuse.

L'évaluation d'un modèle aligné (SFT + DPO/RLHF) est radicalement différente de tout ce que nous avons vu jusqu'à présent. Nous ne cherchons plus à savoir si le modèle a trouvé la "bonne réponse", mais s'il s'est comporté de la manière la plus utile, la plus honnête et la plus sûre possible. C'est le passage de la validation technique à la validation comportementale.

---
## Le déclin des métriques classiques : BLEU, ROUGE et Perplexity

> [!WARNING]
⚠️ **Attention : erreur fréquente ici !** Beaucoup d'étudiants continuent d'utiliser le score BLEU pour évaluer un assistant. C'est une erreur de débutant. Le score BLEU mesure la ressemblance textuelle entre une réponse et une référence humaine. Or, pour une question comme "Comment cuisiner des pâtes ?", il existe un million de bonnes réponses possibles. Un assistant peut donner une recette géniale qui n'a aucun mot commun avec votre texte de référence, et recevoir un score BLEU de zéro. 

Il en va de même pour la **Perplexity**, illustrée en **Figure 12-13 : Prédiction du prochain mot** . 

{{< bookfig src="276.png" week="12" >}}

*   **Explication de la Figure 12-13** : Cette figure montre comment le modèle calcule la probabilité du mot suivant. Une faible perplexité signifie que le modèle n'est pas "surpris" par le texte qu'il lit. 
*   **La limite** : Un modèle peut avoir une perplexité très basse (il prédit très bien le web) tout en étant raciste ou en fournissant des instructions pour fabriquer des explosifs. 

> [!NOTE]
‼ **Je dois insister :** La perplexité mesure la maîtrise de la langue, pas la sagesse de l'assistant.

---
## L'Évaluation Humaine : L'étalon-or (Gold Standard)
Puisque nous alignons les modèles sur les préférences humaines, l'humain reste le juge ultime. Cependant, l'évaluation humaine est lente, coûteuse et sujette à l'humeur. Pour la rendre scientifique, nous utilisons deux approches majeures 👇🏻.

### 1. Les échelles de Likert et le scoring direct
On demande à un expert de noter une réponse de 1 à 5 sur des critères précis. 

> [!IMPORTANT]
🔑 **La règle d'or** : Ne demandez jamais "La réponse est-elle bonne ?". Demandez "À quel point cette réponse est-elle **Helpful** (Utile), **Honest** (Honnête) et **Harmless** (Inoffensive) ?". C'est le triptyque HHH que nous avons vu en 12.1.

### 2. La comparaison par paires (A/B Testing)
Comme nous l'avons vu pour la collecte de données ([**Figure 12-4**]({{< relref "section-12-1.md" >}}#fig-12-4)), les humains sont bien meilleurs pour comparer que pour noter. On présente deux versions de l'IA (Modèle A et Modèle B) et on demande : "Laquelle préférez-vous ?". 

C'est ici qu'intervient la [**Chatbot Arena**](https://arena.ai/). 
*   **Le concept** : C'est un tournoi permanent où des milliers d'utilisateurs discutent avec deux modèles anonymes. 
*   **Le score Elo** : Comme aux échecs, si un petit modèle (ex: Phi-3) bat un grand modèle (ex: GPT-4) dans un duel, son score grimpe en flèche. 
*   🔑 **L'intérêt technique** : Le score Elo de la Chatbot Arena est aujourd'hui considéré comme le reflet le plus fidèle de la "puissance réelle" perçue d'un modèle.

---
## L'Automated Evaluation : Quand l'IA devient le juge
❓ Comment évaluer 1000 versions de votre modèle chaque nuit ?

On ne peut pas réveiller des annotateurs humains à 3h du matin. Nous utilisons donc le concept de **LLM-as-a-judge**. 

Nous utilisons un modèle de référence (le "Juge"), généralement GPT-5 ou Claude 3.6 Sonnet, pour évaluer notre modèle local (Llama-3 ou Phi-3). 
*   **Benchmark MT-Bench** : On pose 80 questions complexes au modèle et le juge note la qualité des réponses sur une échelle de 1 à 10. 
*   **Le biais du juge** : 
>> [!WARNING]
⚠️ Soyez conscients que les modèles juges ont des biais. Ils préfèrent souvent les réponses plus longues (biais de verbosité) et ont tendance à mieux noter les modèles qui leur ressemblent ou qui viennent de la même entreprise.

---
## Bonnes pratiques de collecte de données de préférence
Le succès de votre DPO (Direct Preference Optimization) dépend entièrement de la qualité de vos paires (Chosen/Rejected). Voici les commandements de l'expert :

1.  **La Diversité des thématiques** : Si vous n'utilisez que des paires sur la politesse, votre modèle deviendra très poli mais perdra ses capacités de raisonnement mathématique. Vos paires doivent couvrir le code, la logique, le style créatif et la sécurité.

2.  **L'équilibre des longueurs** : 

>> [!WARNING]
⚠️ **Avertissement** : Si, dans vos exemples "Chosen", la réponse est toujours plus longue que dans la version "Rejected", le modèle va simplement apprendre que "plus long = mieux". 

> Il va se mettre à blablater inutilement (biais de verbosité). Forcez-vous à inclure des exemples où la réponse courte est la version préférée.

3.  **Le "Hard Negative" sémantique** : Une paire de préférence n'est utile que si la différence entre la bonne et la mauvaise réponse est subtile. Si la version rejetée est une suite de lettres aléatoires, le modèle n'apprendra rien. La version rejetée doit être une réponse "presque bonne" mais contenant une petite erreur logique ou un ton légèrement arrogant.

---
## Le piège ultime : La Loi de Goodhart
Nous l'avons évoqué brièvement, mais nous devons y revenir avec force. Regardez la note (*Goodhart's Law*): 

> [!NOTE]
✍🏻 *"Lorsqu'une mesure devient un objectif, elle cesse d'être une bonne mesure."*

Imaginez que vous optimisiez votre modèle uniquement pour obtenir un score de 10/10 sur un benchmark de sécurité. 
*   **Le résultat** : Votre modèle va apprendre à répondre "Je suis désolé, je ne peux pas répondre à cette question" à absolument TOUT, même à "Quelle est la couleur du cheval blanc de Napoléon ?". 
*   **Le diagnostic** : Votre modèle a un score de sécurité parfait, mais une utilité nulle. Vous avez "gagné" le benchmark mais perdu l'utilisateur. 

> [!TIP]
✅ **Mon conseil** : Ne regardez jamais une seule métrique. Suivez toujours la courbe d'utilité (*Helpfulness*) en parallèle de la courbe de sécurité (*Safety*). Si l'une grimpe pendant que l'autre chute, votre alignement est en train de détruire l'intelligence du modèle.

---
## Frontières de la recherche et perspectives
L'alignement est un domaine en pleine explosion. Vous trouverez ci-dessous une liste des pistes passionnantes pour vos futures recherches :

1.  **ORPO (Odds Ratio Preference Optimization)** : C'est une technique encore plus récente qui combine SFT et DPO en **une seule étape**. On ne fait plus deux entraînements séparés. On apprend au modèle à parler et à préférer en même temps. C'est l'avenir de l'efficacité.

2.  **Constitutional AI (IA Constitutionnelle)** : Plutôt que de demander à des humains de classer des milliers de réponses, on donne au modèle une "Constitution" (ex: "Sois utile, ne sois pas raciste"). Le modèle génère ses propres critiques et s'aligne tout seul en suivant ces principes. C'est la méthode utilisée par Anthropic pour le modèle Claude.
3.  **L'Alignement Multimodal** : Comment aligner les préférences d'un modèle qui voit des images (Semaine 10) ? On doit maintenant apprendre à l'IA qu'une image peut être choquante ou trompeuse, et que la description textuelle doit respecter des valeurs humaines même pour le visuel.

---
## Conclusion de la semaine

> [!TIP]
✉️  **Le message final** : Mes chers étudiants, vous avez maintenant toutes les clés. 

Vous savez construire un Transformer, vous savez l'alimenter en données, vous savez le spécialiser par SFT et l'éduquer socialement par DPO. L'alignement est l'étape où l'ingénierie rencontre les sciences humaines. 

N'oubliez jamais : un assistant IA n'est pas un arbitre de vérité. C'est un miroir statistique de ce que nous, humains, avons décidé de valoriser. Soyez fiers de la puissance technologique que vous manipulez, mais soyez humbles devant la responsabilité que représente le fait de définir ce qui est "préférable" pour le reste de l'humanité. 

L'IA de demain sera ce que vous déciderez d'aligner aujourd'hui. 

---
Nous avons terminé notre immense cycle sur l'entraînement et le réglage des modèles ! C'était la partie la plus importante du cours. Respirez. La semaine prochaine, nous allons passer à la mise en production : comment déployer ces géants, comment optimiser leur vitesse pour que l'utilisateur n'attende pas, et comment sécuriser tout cela. Mais avant cela, place au laboratoire de la semaine !