---
title: "Semaine 7 : Clustering et Modélisation de sujets"
weight: 2
bookCollapseSection : true
---

# 📝 Semaine 7 : Clustering et modélisation de sujets

Bonjour à toutes et à tous ! Je suis ravie de vous retrouver. Imaginez un instant que je vous confie les clés d'une archive contenant 50 000 articles scientifiques sur l'IA, sans aucun classement. Par où commenceriez-vous ? 

> [!CAUTION]
🔑 **Je dois insister :** lire tout n'est pas une option.

Aujourd'hui, nous allons apprendre à "prendre de la hauteur" grâce au **Clustering**. Nous allons apprendre à la machine à découvrir d'elle-même la structure cachée des données pour nous dire : "Voici les 20 thèmes majeurs dont parlent vos documents". C'est l'art d'organiser le chaos sémantique.

---

## 📖 Description générale
Cette semaine, nous explorons l'**Apprentissage Non Supervisé** (*Unsupervised Learning*). Nous allons maîtriser le pipeline classique qui transforme une collection de textes bruts en une carte thématique structurée. Nous étudierons en profondeur **BERTopic**, un cadre de travail modulaire qui combine les embeddings (Semaine 6), la réduction de dimension (**UMAP**) et le clustering haute densité (**HDBSCAN**). Enfin, nous verrons comment les LLM peuvent être utilisés comme la "touche finale" pour donner des noms clairs et humains à ces thématiques découvertes par les mathématiques.

---

## 🧠 Pré-requis importants
Pour réussir cette semaine, assurez-vous de maîtriser :
1.  **Manipulation de vecteurs (Semaine 6)** : Savoir transformer du texte en embeddings.
2.  **Pandas & Dataframes** : Être capable de manipuler des colonnes de texte et de labels.
3.  **Statistiques de base** : Comprendre la notion de distribution de mots.

**Ressources pour réviser les pré-requis :**
*   💻 **Documentation Pandas** : [Introduction to DataFrames](https://pandas.pydata.org/docs/user_guide/dsintro.html).
*   📚 **Scikit-learn** : [Introduction aux k-means](https://scikit-learn.org/stable/modules/clustering.html#k-means) (Pour comprendre la différence avec HDBSCAN).
*   🌐 **TensorFlow Projector** : [Visualiser les embeddings en 3D](https://projector.tensorflow.org/) (Outil interactif génial pour l'intuition).

---

## 📚 Ressources utiles pour les concepts de la semaine

### 🌐 Documentations et Articles de référence
*   **Documentation Officielle de BERTopic** : [maartengr.github.io/BERTopic/](https://maartengr.github.io/BERTopic/) – La bible du sujet, créée par Maarten Grootendorst.
*   **Guide UMAP** : [How UMAP Works](https://umap-learn.readthedocs.io/en/latest/how_umap_works.html) – Pour comprendre comment on passe de 768 dimensions à 2 ou 5 dimensions.
*   **Guide HDBSCAN** : [Understanding HDBSCAN](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html) – Pourquoi c'est le meilleur choix pour les données textuelles bruitées.

### 🛠️ Tutoriels pratiques
*   **SBERT Clustering** : [K-Means vs Agglomerative](https://www.sbert.net/examples/applications/clustering/README.html) – Des exemples de code simples pour débuter.
*   **Hugging Face** : [Topic Modeling with BERTopic](https://huggingface.co/blog/bertopic) – Un guide d'implémentation rapide sur le hub.

### 📺 Vidéos recommandées
*   🎥 **Maarten Grootendorst** : [BERTopic for Topic Modeling](https://www.youtube.com/watch?v=uZxQz87lb84) – Présentation claire et concise de BERTopic.

---

> [!TIP]
🔑 **Mon conseil** : Dans cette semaine, vous allez voir des formes colorées apparaître sur vos écrans. Ne vous contentez pas de l'esthétique ! 

> [!WARNING]
⚠️ **Attention :** un bon clustering est un clustering que vous pouvez expliquer. Demandez-vous toujours : "Pourquoi ces deux documents sont-ils dans le même groupe ?".
