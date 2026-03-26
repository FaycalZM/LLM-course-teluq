---
title: "5.4 Applications et limites "
weight: 5
---

## L'IA dans la vie réelle : Au-delà de la curiosité technologique
Bonjour à toutes et à tous ! Nous terminons cette semaine passionnante sur les modèles génératifs. Nous avons vu comment ils sont construits et comment régler leurs paramètres. Mais maintenant, posons-nous la question fondamentale : à quoi cela sert-il vraiment dans le monde professionnel ? Et surtout, quels sont les pièges qui pourraient transformer une innovation brillante en un échec retentissant ? 

> [!IMPORTANT]
🔑 **Je dois insister : un expert en LLM se reconnaît non pas à ce qu'il sait générer, mais à ce qu'il sait anticiper comme erreurs.**

Les modèles *decoder-only*, grâce à leur capacité de généralisation, ne sont plus cantonnés à de simples démonstrations techniques. Ils sont devenus des partenaires de productivité. Cependant, comme nous allons le voir, leur nature probabiliste impose des garde-fous éthiques et techniques rigoureux.

## Domaines d'applications concrètes
Comme l'illustre la [**Figure 1-2**]({{< relref "section-1-1.md" >}}#fig-1-2) et les exemples du [**Tableau 1-1**]({{< relref "section-1-1.md" >}}#tab-1-1), les LLM génératifs redéfinissent plusieurs métiers :

1.  **Assistance à la programmation (Coding)** : C'est sans doute l'application la plus mature. Des modèles comme StarCoder ou GPT-4 ne se contentent pas d'écrire du code ; ils expliquent des algorithmes complexes et aident au débogage. 

>> [!WARNING]
⚠️ **Attention : erreur fréquente ici !** Ne copiez jamais un code généré sans le tester dans un environnement sécurisé (sandbox).

2.  **Rédaction et Créativité** : Du copywriting publicitaire à la scénarisation, les LLM servent de "partenaires de brainstorming". Ils excellent à briser le syndrome de la page blanche en proposant des structures et des idées initiales.
3.  **Résumé et Synthèse** : Donnez un rapport de 50 pages à un modèle avec une grande fenêtre de contexte, et il vous en sortira les 5 points clés en quelques secondes. C'est un gain de temps inestimable pour les professions juridiques et administratives.
4.  **Enseignement et Tutorat** : En paramétrant un persona (vu en [**5.2**]({{< relref "section-5-2.md" >}}#ex-chat-persona)), un LLM peut devenir un tuteur patient qui explique la physique quantique à un enfant de 10 ans.


<a id="hallucinations"></a>

## Le défi de la vérité : Hallucinations et Facticité

> [!IMPORTANT]
🔑 **C'est le concept le plus important de cette section :** Un LLM ne possède pas de modèle interne de la "vérité". Il possède un modèle de la "vraisemblance statistique". 

L'**hallucination** est le phénomène où le modèle génère une réponse fluide, grammaticalement parfaite, mais factuellement fausse. 
*   *Exemple* : Inventer une biographie pour une personne réelle ou citer une loi qui n'existe pas.
*   *Pourquoi ?* Parce que dans l'espace des probabilités, la suite de mots inventée semble "logique" au modèle. 

> [!NOTE]
⚠️ Mes chers étudiants, ne blâmez pas le modèle pour ses hallucinations. C'est sa nature profonde d'imaginer la suite. C'est à vous, concepteurs, de mettre en place des systèmes de vérification.

## Biais, Représentations et Miroirs déformants
Nous l'avons évoqué en semaine 1, mais il est temps d'y revenir avec force. Les LLM sont entraînés sur le web, un endroit magnifique mais aussi rempli de préjugés.
*   **Biais algorithmique** : Si le modèle a lu 90% de textes où les PDG sont des hommes, il aura une probabilité statistique plus élevée de générer "il" pour parler d'un dirigeant d'entreprise.
*   **Toxicité** : Sans l'alignement (RLHF) que nous avons étudié en [**5.2**]({{< relref "section-5-2.md" >}}#RLHF), un modèle pourrait générer des propos haineux ou dangereux simplement parce qu'il les a rencontrés dans ses données d'entraînement.

> [!NOTE]
🔑 **Mon message** : « L'IA n'est pas neutre. Elle est le reflet amplifié de nos propres sociétés. En tant qu'ingénieurs, vous avez le devoir moral d'auditer vos modèles et d'utiliser des techniques de "*Guardrails*" (garde-fous) pour minimiser ces biais. »

## Considérations Éthiques et Légales
Le déploiement de LLM soulève des questions inédites :
1.  **Propriété Intellectuelle** : À qui appartient un poème généré par GPT-4 ? À OpenAI ? À vous ? À personne ? Les tribunaux du monde entier débattent encore de cette question.
2.  **Transparence et AI Act** : L'Union Européenne, via l'AI Act, impose de plus en plus de transparence. Un utilisateur doit savoir s'il interagit avec un humain ou une machine. 

>> [!IMPORTANT]
🔑 **Je dois insister :** La clarté envers l'utilisateur final est la base de la confiance numérique.

3.  **Confidentialité des données** : 
>> [!CAUTION]
⚠️ **Attention !** Tout ce que vous envoyez à une API propriétaire (comme celle d'OpenAI) peut potentiellement être utilisé pour entraîner les futures versions du modèle. Ne partagez jamais de données médicales ou de secrets industriels sans garantie de confidentialité.

## Bonnes pratiques pour un usage responsable
Pour conclure cette semaine, voici vos commandements d'expert :
*   **Human-in-the-loop** : Toujours avoir une révision humaine pour les sorties critiques.
*   **Ancrage (Grounding)** : Ne laissez pas le modèle parler de mémoire ; fournissez-lui des documents sources (nous verrons comment faire avec le RAG en semaine 9).
*   **Température basse pour les faits** : Gardez vos réglages de créativité au minimum pour les tâches sérieuses.

---
Vous avez maintenant une vision complète : vous connaissez la théorie, les réglages, et les responsabilités qui pèsent sur vos épaules. Les LLM sont des outils magiques, mais la magie demande de la discipline. Place maintenant au laboratoire pour mettre en pratique vos talents de pilote de GPT !