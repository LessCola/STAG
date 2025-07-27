import os

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import Embedding

from .loss_func import CosineCommitLoss as CommitLoss
from .text_encoder import TextEncoder

os.environ["TOKENIZERS_PARALLELISM"] = "false"

CLASS_DICT = {
    "agent": "Agent",
    "ai": "Artificial Intelligence",
    "db": "Database",
    "ml": "Machine Learning",
    "hci": "Human-Computer Interaction",
    "diabetes mellitus experimental": "Broad, lab-based studies exploring diabetes mechanisms and treatments.",
    "diabetes mellitus type 1": "Research specifically on autoimmune-related diabetes, primarily in younger patients.",
    "diabetes mellitus type 2": "Research on lifestyle-related diabetes, focusing on prevention, treatment, and management.",
    "cscr": "Cryptography and Security",
    "csdc": "Distributed, Parallel, and Cluster Computing",
    "csit": "Information Theory",
    "csni": "Networking and Internet Architecture",
    "csro": "Robotics",
    "csds": "Data Structures and Algorithms",
    "cshc": "Human-Computer Interaction",
    "cscy": "Computational Geometry",
    "cslg": "Machine Learning",
    "csgt": "Game Theory",
    "cspl": "Programming Languages",
    "cscv": "Computer Vision and Pattern Recognition",
    "csfl": "Formal Languages and Automata Theory",
    "cssi": "Social and Information Networks",
    "cssy": "Systems and Control",
    "cssc": "Symbolic Computation",
    "csai": "Artificial Intelligence",
    "csna": "Numerical Analysis",
    "cscg": "Computational Geometry",
    "cslo": "Logic in Computer Science",
    "cset": "Emerging Technologies",
    "csdl": "Digital Libraries",
    "csir": "Information Retrieval",
    "cscc": "Computational Complexity",
    "csdm": "Discrete Mathematics",
    "csne": "Neural and Evolutionary Computing",
    "csse": "Software Engineering",
    "csma": "Multiagent Systems",
    "cspf": "Performance",
    "csms": "Mathematical Software",
    "csmm": "Multimedia",
    "csoh": "Other Computer Science",
    "cscl": "Computation and Language (Natural Language Processing)",
    "csce": "Computational Engineering, Finance, and Science",
    "csdb": "Databases",
    "cssd": "Social and Information Networks (duplicate of cssi)",
    "csar": "Architecture",
    "csgr": "Graphics",
    "csos": "Operating Systems",
    "csgl": "General Literature",
}

CLASS_EXPLANATION = {
    # * cora
    "agents": "Research on autonomous agents and their interactions in dynamic and distributed environments.",
    "data mining": "The process of extracting meaningful patterns and insights from large datasets.",
    "expert systems": "Studies on computer systems that emulate decision-making abilities of human experts.",
    "games and search": "Research on algorithms and strategies for game playing and problem-solving through search techniques.",
    "knowledge representation": "Studies on structuring and encoding knowledge for reasoning in AI systems.",
    "case based": "Research on learning and problem-solving by leveraging past cases and experiences.",
    "casebased": "Research on learning and problem-solving by leveraging past cases and experiences.",
    "genetic algorithms": "Exploration of optimization techniques inspired by natural evolution and genetics.",
    "neural networks": "Research on computational models inspired by biological neural networks for learning and inference.",
    "probabilistic methods": "Studies on AI methods that use probabilities to handle uncertainty in decision-making.",
    "reinforcement learning": "Research on learning optimal policies through reward-based exploration and exploitation.",
    "rule learning": "Studies on deriving logical rules from data for classification and decision-making.",
    "theory": "Theoretical foundations and principles of machine learning and its algorithms.",
    "nlp": "Research on natural language processing for understanding and generating human language.",
    "planning": "Studies on developing algorithms for automated decision-making and action sequencing.",
    "robotics": "Research on designing and controlling robots for various applications.",
    "speech": "Studies on recognizing, processing, and generating spoken language.",
    "theorem proving": "Research on automated systems to prove mathematical theorems.",
    "vision and pattern recognition": "Research on interpreting visual data and recognizing patterns in images.",
    "computational complexity": "Studies on the difficulty of computational problems and their classification.",
    "computational geometry": "Research on algorithms for geometric problems and data structures.",
    "formal languages": "Studies on mathematical models of computation and language representation.",
    "hashing": "Research on efficient methods for data retrieval using hash functions.",
    "logic": "Studies on formal logic and its application in computational systems.",
    "parallel": "Research on designing algorithms and systems for parallel computation.",
    "quantum computing": "Studies on leveraging quantum mechanics for computational purposes.",
    "randomized": "Research on algorithms that use randomization to improve performance or simplicity.",
    "sorting": "Studies on developing efficient algorithms for ordering data.",
    "concurrency": "Research on techniques for managing simultaneous computations in databases.",
    "deductive": "Studies on logical reasoning and inference in database systems.",
    "object oriented": "Research on object-oriented paradigms for database design and implementation.",
    "performance": "Studies on optimizing the efficiency of database systems.",
    "query evaluation": "Research on algorithms for processing and optimizing database queries.",
    "relational": "Studies on the relational model for database management.",
    "temporal": "Research on managing and querying time-dependent data in databases.",
    "compression": "Studies on methods for reducing the size of data for storage and transmission.",
    "encryption": "Research on techniques to secure data through cryptographic methods.",
    "security": "Studies on protecting systems and data against unauthorized access and threats.",
    "distributed architectures": "Research on designing architectures for distributed computing systems.",
    "high performance computing": "Studies on systems and methods to achieve maximal computational speed.",
    "input output and storage": "Research on optimizing data input, output, and storage mechanisms.",
    "logic design": "Studies on creating digital logic circuits for computational tasks.",
    "memory structures": "Research on the design and optimization of computer memory systems.",
    "microprogramming": "Studies on implementing low-level control logic for computer systems.",
    "vlsi": "Research on designing and fabricating very large-scale integrated circuits.",
    "cooperative": "Studies on systems that support cooperative work and collaboration.",
    "graphics and virtual reality": "Research on visual computing and immersive virtual environments.",
    "interface design": "Studies on designing user interfaces for optimal usability and functionality.",
    "multimedia": "Research on processing and managing multimedia content such as images, video, and audio.",
    "wearable computers": "Studies on designing and using wearable computing devices.",
    "digital library": "Research on systems for organizing and accessing digital content libraries.",
    "extraction": "Studies on extracting structured information from unstructured data.",
    "filtering": "Research on systems to filter information and deliver relevant content to users.",
    "retrieval": "Studies on methods to retrieve relevant data from large collections.",
    "internet": "Research on technologies and protocols that drive internet functionality.",
    "protocols": "Studies on communication rules and standards for networked systems.",
    "routing": "Research on algorithms for determining paths in networks.",
    "wireless": "Studies on wireless communication technologies and networks.",
    "distributed": "Research on distributed computing systems for fault tolerance and scalability.",
    "fault tolerance": "Studies on designing systems that continue functioning despite failures.",
    "memory management": "Research on optimizing memory allocation and utilization in operating systems.",
    "realtime": "Studies on systems designed to perform tasks within strict timing constraints.",
    "compiler design": "Research on designing software that translates high-level programming languages into machine code.",
    "debugging": "Studies on identifying and fixing errors in software systems.",
    "functional": "Research on functional programming paradigms and languages.",
    "garbage collection": "Studies on automated memory management in programming languages.",
    "java": "Research on the Java programming language and its ecosystem.",
    "semantics": "Studies on the meaning and behavior of programs and languages.",
    "software development": "Research on methods and tools to improve software engineering practices.",
    # * citeseer
    "agent": "Research on autonomous agents and their interactions in dynamic and distributed environments.",
    "ai": "Research on developing systems capable of intelligent behavior, such as reasoning, learning, and decision-making.",
    "db": "Studies on database theory, systems, and applications for managing data.",
    "ml": "Research on machine learning algorithms and models for data analysis and prediction.",
    "hci": "Research on human-computer interaction, focusing on designing user interfaces and improving interactions between people and computers.",
    "ir": "Research on information retrieval and search engines for efficient data access.",
    # * pubmed
    "diabetes mellitus experimental": "Experimental studies investigating the biological mechanisms, treatments, and prevention of diabetes.",
    "diabetes mellitus type 1": "Research on autoimmune-related diabetes, focusing on its causes, management, and therapies.",
    "diabetes mellitus type 2": "Research addressing insulin resistance and lifestyle-related diabetes, with an emphasis on prevention and treatment.",
    # * wiki-cs
    "distributed computing architecture": "Articles about systems and models designed for distributing computational tasks across multiple nodes or machines.",
    "databases": "Articles discussing the design, management, and optimization of systems for structured data storage and retrieval.",
    "computer security": "Articles covering topics related to securing computer systems and networks against attacks and unauthorized access.",
    "web technology": "Articles about the development and usage of technologies for creating and managing web-based systems.",
    "computer architecture": "Articles exploring the design and functionality of computer hardware components and systems.",
    "internet protocols": "Articles explaining the standards and conventions that enable communication across computer networks and the internet.",
    "computational linguistics": "Articles discussing computational methods for processing and understanding human languages.",
    "operating systems": "Articles about software that manages hardware resources and provides essential services to applications.",
    "programming language topics": "Articles covering various aspects of programming languages, including syntax, semantics, and applications.",
    "computer file systems": "Articles about methods for storing, organizing, and accessing files on storage devices.",
    # * ogbn-arxiv
    "cscr": "Research on secure communication, cryptographic protocols, and system vulnerabilities.",
    "csdc": "Studies on parallel and distributed systems for improved computation and scalability.",
    "csit": "Research in encoding, transmission, and processing of information across various systems.",
    "csni": "The study of communication networks, internet protocols, and infrastructure design.",
    "csro": "Research on the design, control, and application of robotic systems.",
    "csds": "Research on efficient data structures and algorithms for solving computational problems.",
    "cshc": "Studies on user interface design and improving usability in computer systems.",
    "cscy": "Research on geometric algorithms and their applications in computer science.",
    "cslg": "The development of models and algorithms that enable systems to learn from data.",
    "csgt": "Studies on mathematical modeling and strategic decision-making in competitive environments.",
    "cspl": "Research on programming languages, including their design, implementation, and applications.",
    "cscv": "The study of algorithms and models for visual data analysis and recognition.",
    "csfl": "Research on theoretical frameworks for formal languages and their computational properties.",
    "cssi": "The study of social networks and information sharing in interconnected systems.",
    "cssy": "Research on the design and control of dynamic systems for stability and performance.",
    "cssc": "Studies on computational approaches for solving symbolic and algebraic problems.",
    "csai": "Research on intelligent systems that can perform tasks typically requiring human intelligence.",
    "csna": "Studies on numerical algorithms and their applications in solving mathematical problems.",
    "cscg": "Research on computational methods for geometric modeling and analysis.",
    "cslo": "Studies on formal systems, logic, and their applications in computer science.",
    "cset": "Research on emerging technologies and novel applications in computing.",
    "csdl": "The study of digital library systems for storing, organizing, and retrieving digital content.",
    "csir": "Research on methods for efficient information retrieval and ranking in search systems.",
    "cscc": "Studies on the complexity of computational problems and algorithms.",
    "csdm": "Research on discrete structures and their applications in computer science.",
    "csne": "Studies on neural networks and evolutionary algorithms for optimization and learning.",
    "csse": "Research on software development processes, tools, and methodologies.",
    "csma": "Studies on systems with multiple interacting autonomous agents.",
    "cspf": "Research on performance optimization in computational systems and applications.",
    "csms": "Studies on the development and analysis of mathematical software tools.",
    "csmm": "Research on multimedia systems and technologies for processing and presenting content.",
    "csoh": "Miscellaneous topics in computer science not covered by other categories.",
    "cscl": "Research on natural language processing and computational linguistics.",
    "csce": "Studies on computational methods in engineering, finance, and applied science.",
    "csdb": "Research on database theory, systems, and applications for managing data.",
    "cssd": "Studies on sound, including processing, synthesis, and analysis.",
    "csar": "Research on computer architecture and system-level design.",
    "csgr": "Studies on algorithms and systems for graphics rendering and visualization.",
    "csos": "Research on the design and optimization of operating systems.",
    "csgl": "General topics in computer science, often bridging multiple subfields.",
    # * ogbn-products
    "clothing shoes jewelry": "Products for clothing, shoes, and jewelry.",
    "office products": "Items used in office and professional settings.",
    "books": "Printed or digital books covering various topics and genres.",
    "electronics": "Electronic devices, gadgets, and components.",
    "health personal care": "Products for personal health and hygiene.",
    "sports outdoors": "Equipment and accessories for sports and outdoor activities.",
    "beauty": "Products for personal grooming and skincare.",
    "arts crafts sewing": "Supplies for arts, crafts, and sewing projects.",
    "toys games": "Items designed for play and recreational activities.",
    "tools home improvement": "Tools and equipment for home improvement projects.",
    "cell phones accessories": "Mobile phones and related accessories.",
    "unknown": "Items not classified into a specific category.",
    "movies tv": "Content and media related to films and television.",
    "video games": "Games, consoles, and accessories for gaming.",
    "automotive": "Products for vehicle maintenance and accessories.",
    "cds vinyl": "Physical music formats like CDs and vinyl records.",
    "home kitchen": "Items for home organization and kitchen use.",
    "pet supplies": "Products for pet care and wellbeing.",
    "patio lawn garden": "Products for outdoor living and gardening.",
    "baby products": "Items designed for infants care.",
    "baby": "General products related to babies.",
    "grocery gourmet food": "Food and beverage products, including specialty items.",
    "office school supplies": "Supplies for office or school use.",
    "industrial scientific": "Equipment for industrial and scientific purposes.",
    "musical instruments": "Instruments and accessories for music.",
    "home improvement": "Materials and tools for enhancing homes.",
    "appliances": "Household appliances for various uses.",
    "kitchen dining": "Products for cooking, dining.",
    "amazon fashion": "Apparel and accessories.",
    "all electronics": "A general category for all types of electronics.",
    "software": "Computer programs and applications.",
    "kindle store": "Digital books and resources for Kindle devices.",
    "computers": "Computers, components, and related accessories.",
    "mp3 players accessories": "Portable music players and their accessories.",
    "all beauty": "A comprehensive category for beauty-related products.",
    "gps navigation": "Devices for geographic navigation and tracking.",
    "camera photo": "Cameras and photography accessories.",
    "car electronics": "Electronics designed for vehicles.",
    "collectibles fine art": "Collectible items and fine art pieces.",
    "digital music": "Music available for digital download or streaming.",
    "magazine subscriptions": "Subscriptions to print or digital magazines.",
    "gift cards": "Prepaid cards for gifting or retail use.",
}


class PromptVQ(nn.Module):
    def __init__(
        self,
        args,
        labels,
        categories,
        commit_score=0.25,
        tau_f=0.1,
        emb_model="sentence-transformers/all-mpnet-base-v2",
    ):
        super(PromptVQ, self).__init__()
        self.args = args
        self.top_k = 1

        self.labels = labels
        # sort unique labels
        unique_labels = self.labels.unique()
        unique_labels = unique_labels[unique_labels.sort()[1]].tolist()
        self.label_to_idx = dict(zip(unique_labels, range(len(unique_labels))))
        self.label_to_category = {
            label.item(): category for label, category in zip(self.labels, categories)
        }
        # compute the codebook embeddings check if the label is in the CLASS_EXPLANATION
        text_encoder = TextEncoder(emb_model, args.device)
        codebook_embeddings = []
        for label in unique_labels:
            category = self.label_to_category[label]
            if category in CLASS_EXPLANATION:
                category = CLASS_EXPLANATION[category]
            codebook_embeddings.append(text_encoder.embed(category, "numpy"))
        codebook_embeddings = np.array(codebook_embeddings).squeeze(axis=1)
        # discard the text encoder after use
        del text_encoder
        codebook_vocab = np.array([self.label_to_category[i] for i in unique_labels])
        self.codebook = codebook_vocab
        checkpoint = torch.from_numpy(codebook_embeddings).to("cuda")

        self.num_tokens = checkpoint.shape[0]
        self.embed_dim = checkpoint.shape[1]
        print("Codebook Size: {}".format(self.num_tokens))
        print("Feature Dim: {}".format(self.embed_dim))
        self.tok_embeddings = Embedding(self.num_tokens, self.embed_dim)
        self.tok_embeddings.weight.data = checkpoint
        self.tok_embeddings.weight.data = self.tok_embeddings.weight.data.float()
        self.tok_embeddings.weight.requires_grad = (
            False  # Commented out to allow updates
        )
        self.commit_loss = CommitLoss(
            getattr(self.args, "commit_score_f", commit_score)
        )

        # * Conditional MLP for prompt generation
        # Bottleneck architecture with minimal parameters
        bottleneck_dim = self.embed_dim // 4  # Reduce to 1/4 of original dimension
        self.cond_net = nn.Sequential(
            nn.Linear(self.embed_dim, bottleneck_dim),  # Compress
            nn.ReLU(),
            nn.Linear(bottleneck_dim, self.embed_dim),  # Expand back
        )
        # Initialize weights
        for layer in self.cond_net:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        self.cond_net.to(self.args.device)

        # * compute the class similarity weights based on the codebook embeddings
        self.class_similarity_weights = torch.matmul(
            self.tok_embeddings.weight, self.tok_embeddings.weight.T
        )
        self.class_similarity_weights = (
            self.class_similarity_weights
            / self.class_similarity_weights.norm(dim=1, keepdim=True)
        )
        self.class_similarity_weights.requires_grad = False
        self.tau_f = getattr(self.args, "tau_f", tau_f)

    def update_codebook(self, few_shot_train_mask):
        unique_labels = self.labels[few_shot_train_mask].unique()
        unique_labels = unique_labels[unique_labels.sort()[1]].tolist()
        # print(f"Unique labels: {unique_labels}")
        label_idx = [self.label_to_idx[label] for label in unique_labels]
        self.codebook = self.codebook[label_idx]
        # IndexError: boolean index did not match indexed array along dimension 0; dimension is 7 but corresponding boolean dimension is 2
        self.tok_embeddings.weight.data = self.tok_embeddings.weight.data[label_idx]
        self.class_similarity_weights = torch.matmul(
            self.tok_embeddings.weight, self.tok_embeddings.weight.T
        )
        self.class_similarity_weights = (
            self.class_similarity_weights
            / self.class_similarity_weights.norm(dim=1, keepdim=True)
        )
        self.class_similarity_weights.requires_grad = False
        # create a dictionary to map the original labels to the new labels
        self.label_to_idx = dict(zip(unique_labels, range(len(unique_labels))))
        for layer in self.cond_net:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

    def search_torch(self, z_e):
        """
        Finds the top-k nearest neighbors using cosine similarity.

        Args:
            z_e (torch.Tensor): Input tensor of shape [batch_size, z_dim].

        Returns:
            tuple: A tuple of similarities and indices of the nearest neighbors.
        """
        z_flattened = z_e.view(z_e.shape[0], -1)
        tok_embeddings_weight = self.tok_embeddings.weight

        # Normalize the input tensor and the embedding weights
        z_flattened_norm = z_flattened / z_flattened.norm(dim=1, keepdim=True)
        tok_embeddings_weight_norm = tok_embeddings_weight / tok_embeddings_weight.norm(
            dim=1, keepdim=True
        )

        # Compute the cosine similarity
        cosine_similarity = torch.einsum(
            "bd,dn->bn",
            z_flattened_norm,
            rearrange(tok_embeddings_weight_norm, "n d -> d n"),
        )

        # Get the top-k highest similarities
        _, max_indices = torch.topk(cosine_similarity, self.top_k, dim=1, largest=True)

        return max_indices, tok_embeddings_weight, cosine_similarity

    def quantize_torch(self, z_e, prompt=True):
        # combine prompt and z_e with element-wise multiplication
        z_e = z_e * self.cond_net(z_e) if prompt else z_e

        # search using PyTorch
        min_indices, tok_embeddings_weight, cosine_similarity = self.search_torch(z_e)
        tokens = self.get_token(min_indices)

        # Convert indices back to torch tensors
        z_q = tok_embeddings_weight[min_indices].squeeze(1)  # Shape [batch_size, z_dim]

        return z_q, tokens, cosine_similarity

    def forward(self, z_e, inference=False):
        z_q, tokens, cosine_similarity = self.quantize_torch(z_e)
        if inference:
            return (
                z_q,
                tokens,
            )  # cosine_similarity
        loss = self.commit_loss(z_e, z_q)
        z_q = z_e + (z_q - z_e).detach()
        return loss, z_q, tokens, cosine_similarity

    def get_token(self, min_indices):
        if isinstance(min_indices, torch.Tensor):
            min_indices = min_indices.cpu().numpy()
        texts = []
        for indices in min_indices:
            texts.append(self.codebook[indices])
        return texts
