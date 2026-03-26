---
title: "Semaine 8 : Ingénierie des prompts"
weight: 3
---

# 📝 Sommaire de la Semaine 8 : Ingénierie des prompts

Bonjour à toutes et à tous ! Je suis ravie de vous retrouver pour entamer cette semaine. Jusqu'ici, nous avons construit des moteurs et cartographié des données. Aujourd'hui, nous allons apprendre à "murmurer à l'oreille des IA". 

> [!IMPORTANT]
🔑 **Je dois insister :** le **Prompt Engineering** n'est pas une simple astuce de rédaction, c'est une véritable discipline d'ingénierie qui permet de débloquer les capacités de raisonnement les plus profondes des LLM. 

Préparez-vous à transformer un simple outil de discussion en un expert capable de résoudre des énigmes complexes étape par étape.

---

## 📖 Description générale
Cette semaine est dédiée à l'**art et la science du prompt**. Nous allons décomposer l'anatomie d'un prompt efficace (Persona, Contexte, Instructions, Format). Nous explorerons les techniques avancées qui font la différence entre une réponse banale et un raisonnement brillant : le **Few-shot learning** (apprendre par l'exemple), le **Chain-of-Thought** (penser étape par étape) et le **Tree-of-Thought** (explorer plusieurs chemins). Enfin, nous verrons comment garder le contrôle sur la machine grâce au **Constrained Sampling** pour forcer l'IA à respecter des structures strictes comme le JSON.

---

## 🧠 Pré-requis importants
Pour cette semaine, vous devez avoir bien assimilé :
1.  **Intuition des modèles génératifs (Semaine 5)** : Comprendre comment le modèle prédit le prochain token.
2.  **Paramètres de génération (Semaine 5)** : Savoir ce que sont la température et le Top-P.
3.  **Bases du Python** : Pour manipuler des chaînes de caractères et faire des appels aux modèles via `transformers` ou `llama-cpp-python`.

**Ressources pour réviser les pré-requis :**
*   💻 **Tutoriel Hugging Face** : [Pipeline de génération de texte](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.TextGenerationPipeline) .
*   📚 **Documentation Python** : [F-strings et manipulation de chaînes](https://docs.python.org/3/tutorial/inputoutput.html) .

---

## 📚 Ressources utiles pour les concepts de la semaine

### 🌐 Documentations et Guides de référence (Vérifiés)
*   **PromptingGuide.ai** : [The Prompt Engineering Guide](https://www.promptingguide.ai/fr) – La ressource la plus complète et multilingue sur le sujet .
*   **Hugging Face Blog** : [LLM Prompting Guide](https://huggingface.co/docs/transformers/v4.49.0/tasks/prompting) – Un excellent point de départ technique .
*   **Documentation OpenAI** : [Prompt Engineering Best Practices](https://platform.openai.com/docs/guides/prompt-engineering) – Des conseils applicables à tous les modèles, pas seulement GPT .

### 📺 Vidéos recommandées (Vérifiées)
*   🎥 **DeepLearning.AI (Andrew Ng)** : [ChatGPT Prompt Engineering for Developers](https://www.youtube.com/watch?v=H4YK_7MAckk) – Un cours fondamental sur la structuration des instructions .
*   🎥 **Elvis Saravia** : [Advanced Prompting Techniques](https://www.youtube.com/watch?v=dOxUroR57xs) – Focus sur le raisonnement et les agents .

### 🛠️ Outils de contrôle
*   **llama-cpp-python** : [Guide sur les grammaires GBNF](https://github.com/abetlen/llama-cpp-python#json-schema-and-grammar-based-sampling) – Pour forcer les modèles locaux à sortir du JSON .

---

> [!TIP]
🔑 Dans cette semaine, vous allez devenir des "sculpteurs de contexte". 

> [!WARNING]
⚠️ **Attention :** un prompt trop long peut perdre le modèle au milieu du texte. C'est ce qu'on appelle le phénomène du "Lost in the Middle". Apprenez à être précis, concis et structuré !
