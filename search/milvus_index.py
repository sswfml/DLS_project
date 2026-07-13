from pymilvus import (
    connections, Collection, CollectionSchema, 
    FieldSchema, DataType, utility
)
import numpy as np
from pathlib import Path
import time
import json

class MilvusSearch:
    """Milvus-based vector search implementation"""
    
    def __init__(self, collection_name, dimension, index_type='IVF_PQ', 
                 host='localhost', port='19530', **kwargs):
        self.collection_name = collection_name
        self.dimension = dimension
        self.index_type = index_type
        self.host = host
        self.port = port
        self.kwargs = kwargs
        
        self.collection = None
        self._connect()
        self._create_collection()
    
    def _connect(self):
        """Connect to Milvus server"""
        connections.connect(
            alias="default",
            host=self.host,
            port=self.port
        )
        print(f"Connected to Milvus at {self.host}:{self.port}")
    
    def _create_collection(self):
        """Create collection if it doesn't exist"""
        # Check if collection exists
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            print(f"Loaded existing collection: {self.collection_name}")
            return
        
        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
            FieldSchema(name="track_id", dtype=DataType.INT64),
            FieldSchema(name="genre", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="artist", dtype=DataType.VARCHAR, max_length=100),
        ]
        
        schema = CollectionSchema(fields, description="Music embeddings collection")
        self.collection = Collection(self.collection_name, schema)
        print(f"Created new collection: {self.collection_name}")
        
        # Create index
        self._create_index()
    
    def _create_index(self):
        """Create index for the collection"""
        index_params = {
            "metric_type": "L2",
            "index_type": self.index_type,
            "params": {}
        }
        
        if self.index_type == "IVF_PQ":
            index_params["params"] = {
                "nlist": self.kwargs.get('nlist', 4096),
                "m": self.kwargs.get('m', 16),
                "nbits": self.kwargs.get('nbits', 8)
            }
        elif self.index_type == "HNSW":
            index_params["params"] = {
                "M": self.kwargs.get('M', 16),
                "efConstruction": self.kwargs.get('ef_construction', 200)
            }
        else:
            # Default: IVF_FLAT
            index_params["params"] = {
                "nlist": self.kwargs.get('nlist', 4096)
            }
        
        self.collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        print(f"Created {self.index_type} index")
    
    def add(self, embeddings, metadata=None):
        """Add vectors to collection"""
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
        
        # Prepare data
        data = [embeddings.tolist()]
        
        # Add metadata fields if available
        if metadata:
            for field in ['track_id', 'genre', 'title', 'artist']:
                if field in metadata:
                    data.append(metadata[field])
        
        # Insert data
        start_time = time.time()
        result = self.collection.insert(data)
        insert_time = time.time() - start_time
        
        # Flush to ensure data is persisted
        self.collection.flush()
        
        print(f"Added {len(embeddings)} vectors in {insert_time:.2f}s")
        return result
    
    def search(self, query, k=10, params=None):
        """Search for nearest neighbors"""
        if len(query.shape) == 1:
            query = query.reshape(1, -1)
        
        # Load collection to memory
        self.collection.load()
        
        # Search parameters
        search_params = {
            "metric_type": "L2",
            "params": params or {}
        }
        
        if self.index_type == "IVF_PQ":
            search_params["params"]["nprobe"] = params.get('nprobe', 20)
        
        # Perform search
        start_time = time.time()
        results = self.collection.search(
            data=query.tolist(),
            anns_field="embedding",
            param=search_params,
            limit=k,
            output_fields=["track_id", "genre", "title", "artist"]
        )
        search_time = time.time() - start_time
        
        # Format results
        formatted_results = []
        for hits in results:
            query_results = []
            for hit in hits:
                result = {
                    'id': hit.id,
                    'distance': hit.distance,
                    'track_id': hit.entity.get('track_id'),
                    'genre': hit.entity.get('genre'),
                    'title': hit.entity.get('title'),
                    'artist': hit.entity.get('artist')
                }
                query_results.append(result)
            formatted_results.append(query_results)
        
        return formatted_results, search_time
    
    def delete(self):
        """Delete the collection"""
        utility.drop_collection(self.collection_name)
        print(f"Deleted collection: {self.collection_name}")
    
    def get_stats(self):
        """Get collection statistics"""
        self.collection.load()
        return {
            'num_entities': self.collection.num_entities,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'collection_name': self.collection_name
        }
