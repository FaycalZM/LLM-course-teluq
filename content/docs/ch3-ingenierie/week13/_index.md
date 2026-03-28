---
title: "Semaine 13 : Déploiement, optimisation et éthique"
weight: 3
bookCollapseSection : true
---


# 📝 Semaine 13 : Déploiement, optimisation et éthique

Bonjour à toutes et à tous ! Nous approchons de la ligne d'arrivée. Jusqu'ici, nous avons vécu dans le confort de nos notebooks et de nos environnements de recherche. Mais aujourd'hui, le monde réel nous appelle. 

🔑 **Je dois insister :** un modèle de langage n'a de valeur que s'il est utile, rapide et, par-dessus tout, responsable. Déployer un LLM, ce n'est pas juste "cliquer sur un bouton", c'est garantir qu'il répondra en millisecondes sans compromettre la sécurité des données ou l'éthique de votre entreprise. Préparez-vous à transformer vos prototypes en solutions de production robustes !

---
### 📖 Description générale
Cette semaine traite de la transition critique entre le laboratoire et la production. Nous explorerons les mécaniques d'**inférence optimisée** (KV cache, quantification post-entraînement) pour réduire la latence et les coûts. Nous aborderons ensuite le volet crucial de la **Sécurité** (détection d'injections de prompts, jailbreaks) et de la **Légalité** (AI Act européen, RGPD). Enfin, nous définirons les **Bonnes pratiques de déploiement** : monitoring, logging et tests A/B pour assurer une amélioration continue de vos assistants IA.

---

### 🧠 Pré-requis importants
Pour réussir cette phase finale, vous devez maîtriser :
1.  **Mécanique des Transformers (Semaine 3)** : Essentiel pour comprendre pourquoi le KV cache est nécessaire.
2.  **Quantification (Semaine 11)** : Savoir faire la différence entre NF4 et les formats d'inférence (GGUF, AWQ).
3.  **Bases du Web & API** : Comprendre comment une requête circule entre un client et un serveur.

**Ressources pour réviser les pré-requis :**
*   💻 **Hugging Face Blog** : [A Guide to Quantization](https://huggingface.co/blog/introduction-to-quantization) .
*   📚 **Maarten Grootendorst** : [Visual Guide to Quantization](https://maartengr.github.io/2023/10/21/visual-guide-to-quantization.html) .

---

### 📚 Ressources utiles pour les concepts de la semaine

#### 🌐 Articles et Blogs de référence
*   **Hugging Face Blog** : [Responsible AI at Hugging Face](https://huggingface.co/blog/responsible-ai) – Un manifeste sur les biais et la transparence.
*   **Llama.cpp GitHub** : [GGUF Format & Inference](https://github.com/ggerganov/llama.cpp) – Le projet leader pour l'inférence locale optimisée.
*   **AI Act Official** : [EU Artificial Intelligence Act Info](https://artificialintelligenceact.eu/) – Pour comprendre le cadre réglementaire européen.

#### 📺 Vidéos recommandées
*   🎥 **Andrej Karpathy** : [Intro to LLMs (Security & Future)](https://www.youtube.com/watch?v=zjkBMFhNj_g) – (À partir de 28:00 pour les attaques par injection et les enjeux de sécurité).
*   🎥 **NVIDIA** : [Optimizing LLM Inference](https://www.youtube.com/watch?v=fS_0W_Vp4sM) – Détails techniques sur le throughput et la latence.

#### 🛠️ Frameworks de production
*   **vLLM Documentation** : [High-throughput serving](https://docs.vllm.ai/en/latest/) – La bibliothèque de référence pour servir des LLM à grande échelle.
*   **Guardrails AI** : [Input/Output Validation](https://www.guardrailsai.com/docs) – Pour sécuriser les interactions.

---

🔑 **Mon conseil** : En production, la vitesse est une forme de politesse. 

⚠️ **Attention :** un utilisateur n'attendra jamais 30 secondes pour une réponse. Apprenez à sacrifier un peu de précision pour beaucoup de réactivité. C'est là que l'ingénieur prend le pas sur le chercheur !
