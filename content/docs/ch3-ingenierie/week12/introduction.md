---
title: "Introduction"
weight: 1
---

## De l'instruction au jugement humain : L'alignement des LLM

Bonjour à toutes et à tous ! Je suis ravie de vous retrouver pour cette semaine, qui marque l'aboutissement éthique et technique de notre parcours. Jusqu'ici, nous avons appris à remplir le cerveau de notre IA de connaissances et à lui apprendre à obéir à des ordres. Mais aujourd'hui, nous passons de la "tête" au "cœur" : nous allons apprendre à donner des valeurs à la machine. 
> [!IMPORTANT]
🔑 **Je dois insister :** un modèle savant qui n'est pas aligné est comme un génie sans boussole ; il peut être aussi dangereux qu'utile. 

Préparez-vous à découvrir comment nous transformons un algorithme en un assistant digne de confiance, capable de refléter la subtilité du jugement humain.

---
**Rappel semaine précédente** : La semaine dernière, nous avons maîtrisé le Fine-tuning supervisé (SFT) et la méthode QLoRA, apprenant à transformer un modèle de base en un assistant capable de suivre des instructions précises tout en optimisant l'usage de la mémoire GPU.

**Objectifs de la semaine** :
À la fin de cette semaine, vous saurez :
> *   Expliquer pourquoi le Fine-tuning supervisé (SFT) est insuffisant pour créer un assistant sûr et agréable.
> *   Comprendre le fonctionnement du RLHF (Reinforcement Learning from Human Feedback) et le rôle du modèle de récompense.
> *   Détailler l'architecture et les avantages mathématiques du DPO (Direct Preference Optimization).
> *   Identifier les biais potentiels introduits lors de la phase d'alignement.
> *   Implémenter un entraînement DPO sur des paires de préférences avec la bibliothèque TRL.
