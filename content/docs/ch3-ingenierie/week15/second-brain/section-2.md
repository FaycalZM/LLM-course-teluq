---
title: "2. La Forge du Spécialiste"
weight: 2
---

# Section 2 : La Forge du Spécialiste – Distillation et Fine-tuning

Bonjour à toutes et à tous ! Quel plaisir de vous retrouver. Dans la section précédente, nous avons construit le réservoir de notre assistant : une base de données vectorielle propre et sécurisée. Mais un réservoir, sans moteur pour transformer l'énergie, n'est qu'un objet inerte. Aujourd'hui, nous montons d'un étage dans la complexité : nous allons entrer dans la forge. 

> [!IMPORTANT]
**Je dois insister :** utiliser un modèle "tout-venant" pour résumer vos pensées intimes est une erreur. Pour que l'assistant comprenne votre style et la structure de vos notes, il lui faut une éducation sur mesure. Aujourd'hui, nous allons voir comment le projet **Second Brain** utilise la puissance des géants (GPT-4o) pour éduquer des modèles plus petits (Llama-3.1) via la **distillation** et le **fine-tuning efficace**. Respirez, nous allons donner du caractère à votre IA !

---

## 2.1 La Génération de Dataset par Distillation

Une question doit vous hanter : si nous voulons que notre assistant sache résumer nos notes, où allons-nous trouver des milliers d'exemples de "Notes -> Résumé parfait" pour l'entraîner ? 

En entreprise, ces données n'existent presque jamais. 

Le projet Second Brain résout ce problème par la **Distillation**, une technique que nous avons abordée en **Section 11.4**. Ici, le "Professeur" (GPT-4o) va corriger les copies de l' "Élève" (Llama-3.1).

### A. Le Pipeline de Création

Regardons ensemble la Figure 15-5.

{{< bookfig src="15_5.png" week="15" >}}

**Explication de la Figure** : 
Cette illustration décrit un cycle de production de données de haute qualité :
1.  **Extract Documents** : On puise dans les documents validés par notre juge de qualité (Section 1).
2.  **Document Summarization** : C'est l'étape clé. On envoie ces documents à GPT-4o. Mais notez la boucle : on change les paramètres de génération (Température, Top-K) pour obtenir des variantes. 
3.  **Summary Filtering** : On ne garde que les meilleurs résumés. 
4.  **Push Dataset** : Le résultat est envoyé vers le **Data Registry** (Hugging Face Hub).


**Mon intuition** : La distillation consiste à transférer le "raisonnement" d'un modèle massif vers un modèle compact. Le petit modèle n'apprend pas seulement les faits, il apprend la *manière* dont le grand modèle synthétise l'information.

### B. Implémentation du Générateur
Le fichier `src/second_brain_offline/application/dataset/generators.py` contient la logique de cette forge.

```python
# [SOURCE: src/second_brain_offline/application/dataset/generators.py#L93]
def __augmented_summarization_loop(self, documents: list[Document], loops: int = 3) -> list[Document]:
    # On utilise un agent de synthèse (Semaine 5)
    summarization_agent = SummarizationAgent(
        max_characters=self.summarization_max_characters,
        model_id=self.summarization_model, # Souvent gpt-4o-mini pour le coût
        max_concurrent_requests=self.max_workers
    )
    # ...
    for i in range(loops):
        # On fait varier la température (Semaine 5.3) pour augmenter la diversité
        temperature = i * 0.5 / loops
        summarized_documents = summarization_agent(copied_documents, temperature=temperature)
        # ...
```

Observez l'usage de la température variable. 
> [!IMPORTANT]
**Je dois insister :** si vous générez des données d'entraînement avec une température de 0, votre modèle final sera un robot monotone. 

> En injectant un peu de hasard maîtrisé dans votre dataset de distillation, vous donnez de la "souplesse sémantique" à votre futur assistant.

---

## 2.2 Fine-tuning avec Unsloth & LoRA : La Vitesse Pure

Une fois le dataset prêt, nous passons à l'entraînement réel. Le projet utilise **Unsloth**, une bibliothèque qui optimise les noyaux (kernels) de calcul de PyTorch pour rendre le fine-tuning jusqu'à 2 fois plus rapide.

### A. L'Architecture d'Entraînement
La [**Figure 15-4**]({{< relref "section-1.md" >}}#fig-15-4) est le tableau de bord de votre GPU T4.

**Explication** :
*   **Apply (Q)LoRA** : C'est la mise en pratique de la **Section 11.2**. On ne touche pas au 8 milliards de paramètres de Llama-3.1. On y injecte des matrices de bas rang. 
*   **Tweak & Loop** : Le schéma montre une boucle de rétroaction. Si l'évaluation montre que le modèle commence à "halluciner" (Section 5.4), on ajuste les hyperparamètres et on recommence. 
*   **Model Registry** : Une fois le modèle "bon", on sauvegarde l'adaptateur LoRA séparément.

> [!NOTE]
**Le secret de l'économie** : Grâce à QLoRA (Section 11.3), le projet permet de fine-tuner Llama-3.1 8B sur une simple carte T4 de 16 Go de VRAM. Sans cette technique, il nous faudrait un cluster de serveurs hors de prix.


### B. Le code du Fine-tuning
Dans le notebook `finetuning.ipynb`, la configuration de LoRA est exemplaire de rigueur industrielle :

```python
# [SOURCE: src/second_brain_offline/application/models/finetuning.ipynb]
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-instruct-bnb-4bit", # 4-bit loading (Section 11.3)
    max_seq_length = 2048,
    load_in_4bit = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Le Rang (Section 11.2)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
)
```

> [!WARNING]
**Attention : erreur fréquente ici !** Notez que Karpathy dans minGPT (Projet 1) n'utilisait pas LoRA. Ici, nous l'utilisons car nous travaillons sur un modèle **pré-entraîné**. 


**La règle d'or** : On entraîne `minGPT` pour comprendre le langage ; on fine-tune `Second Brain` pour comprendre une mission.

---

## 2.3 Le Suivi d'Expérience avec Comet ML

"Comment savoir si votre entraînement de cette nuit a été meilleur que celui d'hier ?" 

En production, on n'utilise pas des fichiers texte pour noter les résultats. On utilise un **Experiment Tracker**.

Le projet intègre **Comet ML**. 
*   **Monitoring** : À chaque itération, la courbe de "Loss" (Perte) est envoyée sur un tableau de bord web.
*   **Comparaison** : Vous pouvez comparer visuellement l'impact de changer le `learning_rate` de `2e-4` à `1e-5` (Section 11.4).

> [!IMPORTANT]
**Je dois insister :** Un ingénieur qui ne monitore pas ses courbes est un pilote qui vole sans instruments dans le brouillard. Le projet Second Brain vous force à adopter cette rigueur professionnelle.

---

## 2.4 Le Déploiement "Serverless" : Hugging Face Endpoints

Une fois le modèle forgé, il faut le rendre accessible à notre application web. Le projet utilise une approche moderne : les **Hugging Face Inference Endpoints**.

### A. La logique d'appel
Le fichier `call_huggingface_dedicated_endpoint.py` montre comment interroger votre modèle final.

```python
# [SOURCE: tools/call_huggingface_dedicated_endpoint.py]
client = OpenAI(
    base_url=settings.HUGGINGFACE_DEDICATED_ENDPOINT,
    api_key=settings.HUGGINGFACE_ACCESS_TOKEN,
)

chat_completion = client.chat.completions.create(
    model="tgi", # Utilisation du Text Generation Inference (Semaine 13.1)
    messages=[{"role": "user", "content": prompt}],
    stream=True, # Le "Streaming" pour la latence (Semaine 13.1)
)
```

**Analyse de l'expert** : Notez que bien que le modèle soit sur Hugging Face, nous utilisons le client `OpenAI`. Pourquoi ? Parce que le standard **TGI** (Hugging Face) est désormais compatible avec l'API OpenAI. 

**C'est une leçon d'interopérabilité capitale** pour votre carrière : apprenez les standards, pas seulement les bibliothèques.

---

## 2.5 Éthique de la Forge : Le Biais de la Distillation

> [!CAUTION]
Mes chers étudiants, soyez conscients de l'effet d'écho. 

Quand vous distillez le savoir de GPT-4o vers Llama-3.1 :
1.  **L'amplification des préjugés** : Si GPT-4o a un biais subtil dans sa façon de résumer les opinions politiques ou sociales, Llama-3.1 va mémoriser ce biais comme étant la "règle d'or". 
2.  **L'illusion de la certitude** : Les modèles distillés ont tendance à être plus affirmatifs que leurs professeurs, perdant parfois les nuances de doute ("Il semble que...", "Peut-être...").
3.  **L'empreinte carbone** : Le fine-tuning consomme de l'énergie. 

**Mon conseil** : Le projet montre que l'on peut limiter cela en utilisant des modèles quantifiés et des adaptateurs LoRA. La sobriété numérique commence par l'optimisation des hyperparamètres.

---

## Synthèse

Vous avez vu comment nous prenons la donnée brute de la Section 1 pour forger un expert. Nous avons utilisé GPT-4o comme maître d'école et LoRA comme instrument de chirurgie.

> [!TIP]
**Le message à retenir** : Le fine-tuning n'est pas une fin en soi, c'est une spécialisation. Votre assistant sait désormais comment VOUS voulez qu'il résume vos notes. 


Dans la section prochaine, nous allons enfin assembler le tout. Nous allons voir comment l'agent utilise ce modèle spécialisé et ses outils de recherche pour devenir une IA **active** et **auditable**. Prêts pour le grand final de l'orchestration ?