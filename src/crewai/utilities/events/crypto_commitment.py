#!/usr/bin/env python3
"""
CRYPTO COMMITMENT PROTOCOL - Cryptographic Agent Coordination
English word commitments with escrow validation for Byzantine fault tolerance
Each agent commits to their work with an encrypted word, escrow validates completion
"""

import time
import redis
import json
import hashlib
import secrets
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend

class CommitmentStatus(Enum):
    PENDING = "pending"
    COMMITTED = "committed"
    EXECUTING = "executing"
    REVEALING = "revealing"
    VALIDATED = "validated"
    FAILED = "failed"

@dataclass
class AgentCommitment:
    agent_id: str
    task_id: str
    commitment_word: str
    encrypted_commitment: bytes
    public_key: bytes
    timestamp: float
    status: CommitmentStatus
    assignment_data: Dict[str, Any]

@dataclass
class TransactionRecord:
    transaction_id: str
    task_id: str
    participants: List[str]
    commitments: Dict[str, AgentCommitment]
    start_time: float
    completion_time: Optional[float] = None
    success: bool = False
    validation_proof: Optional[str] = None

class CryptoCommitmentAgent:
    """Agent with cryptographic commitment capabilities"""
    
    def __init__(self, agent_id: str, redis_client):
        self.agent_id = agent_id
        self.r = redis_client
        
        # Generate RSA key pair for this agent
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        self.public_key = self.private_key.public_key()
        
        # Word pool for commitments (6+ characters)
        self.word_pool = [
            "elephant", "telescope", "mountain", "rainbow", "keyboard", "lightning",
            "butterfly", "adventure", "wonderful", "gorgeous", "brilliant", "treasure",
            "paradise", "symphony", "chocolate", "fireworks", "fantastic", "incredible",
            "mysterious", "beautiful", "dangerous", "important", "somewhere", "everyone",
            "computer", "internet", "software", "programming", "development", "coordination"
        ]
        
        print(f"🔐 CRYPTO AGENT INITIALIZED: {self.agent_id}")
    
    def generate_commitment_word(self) -> str:
        """Generate random commitment word from pool"""
        return secrets.choice(self.word_pool)
    
    def create_commitment(self, task_id: str, assignment_data: Dict[str, Any]) -> AgentCommitment:
        """Create cryptographic commitment for task assignment"""
        
        # Generate commitment word
        commitment_word = self.generate_commitment_word()
        
        # Encrypt commitment word with our private key (sign it)
        commitment_bytes = commitment_word.encode('utf-8')
        encrypted_commitment = self.private_key.sign(
            commitment_bytes,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        # Serialize public key for verification
        public_key_bytes = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        commitment = AgentCommitment(
            agent_id=self.agent_id,
            task_id=task_id,
            commitment_word=commitment_word,  # Keep privately for later reveal
            encrypted_commitment=encrypted_commitment,
            public_key=public_key_bytes,
            timestamp=time.time(),
            status=CommitmentStatus.COMMITTED,
            assignment_data=assignment_data
        )
        
        print(f"🔒 COMMITMENT CREATED: {self.agent_id} → '{commitment_word}' for {task_id}")
        return commitment
    
    def reveal_commitment(self, commitment: AgentCommitment) -> str:
        """Reveal commitment word after task completion"""
        
        print(f"🔓 REVEALING COMMITMENT: {self.agent_id} → '{commitment.commitment_word}'")
        return commitment.commitment_word
    
    def validate_other_commitment(self, commitment: AgentCommitment, revealed_word: str) -> bool:
        """Validate another agent's commitment using their public key"""
        
        try:
            # Load their public key
            public_key = serialization.load_pem_public_key(
                commitment.public_key,
                backend=default_backend()
            )
            
            # Verify the signature
            public_key.verify(
                commitment.encrypted_commitment,
                revealed_word.encode('utf-8'),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            print(f"✅ COMMITMENT VALID: {commitment.agent_id} revealed '{revealed_word}'")
            return True
            
        except Exception as e:
            print(f"❌ COMMITMENT INVALID: {commitment.agent_id} - {e}")
            return False

class CryptoEscrowAgent:
    """Escrow agent that validates cryptographic commitments"""
    
    def __init__(self, redis_client):
        self.r = redis_client
        self.active_transactions = {}
        
        print("🏛️ CRYPTO ESCROW AGENT INITIALIZED")
        print("   Ready to validate agent commitments")
    
    def start_transaction(self, task_id: str, participants: List[str]) -> str:
        """Start new transaction with participant commitments"""
        
        transaction_id = f"tx_{int(time.time() * 1000)}_{hashlib.md5(task_id.encode()).hexdigest()[:8]}"
        
        transaction = TransactionRecord(
            transaction_id=transaction_id,
            task_id=task_id,
            participants=participants,
            commitments={},
            start_time=time.time()
        )
        
        self.active_transactions[transaction_id] = transaction
        
        # Store in Redis for persistence
        self.r.hset(
            f"crypto_escrow:transactions:{transaction_id}",
            mapping={
                'task_id': task_id,
                'participants': json.dumps(participants),
                'start_time': str(transaction.start_time),
                'status': 'started'
            }
        )
        
        print(f"🚀 TRANSACTION STARTED: {transaction_id}")
        print(f"   Task: {task_id}")
        print(f"   Participants: {participants}")
        
        return transaction_id
    
    def receive_commitment(self, transaction_id: str, commitment: AgentCommitment) -> bool:
        """Receive and store agent commitment"""
        
        if transaction_id not in self.active_transactions:
            print(f"❌ UNKNOWN TRANSACTION: {transaction_id}")
            return False
        
        transaction = self.active_transactions[transaction_id]
        
        if commitment.agent_id not in transaction.participants:
            print(f"❌ UNAUTHORIZED PARTICIPANT: {commitment.agent_id}")
            return False
        
        # Store commitment
        transaction.commitments[commitment.agent_id] = commitment
        
        # Update Redis
        commitment_data = {
            'agent_id': commitment.agent_id,
            'encrypted_commitment': commitment.encrypted_commitment.hex(),
            'public_key': commitment.public_key.decode('utf-8'),
            'timestamp': str(commitment.timestamp),
            'assignment_data': json.dumps(commitment.assignment_data)
        }
        
        self.r.hset(
            f"crypto_escrow:commitments:{transaction_id}:{commitment.agent_id}",
            mapping=commitment_data
        )
        
        print(f"📝 COMMITMENT RECEIVED: {commitment.agent_id} for {transaction_id}")
        return True
    
    def validate_transaction_completion(self, transaction_id: str, 
                                      revealed_words: Dict[str, str]) -> bool:
        """Validate all commitments match revealed words"""
        
        if transaction_id not in self.active_transactions:
            print(f"❌ UNKNOWN TRANSACTION: {transaction_id}")
            return False
        
        transaction = self.active_transactions[transaction_id]
        
        print(f"🔍 VALIDATING TRANSACTION: {transaction_id}")
        print(f"   Commitments: {len(transaction.commitments)}")
        print(f"   Reveals: {len(revealed_words)}")
        
        # Check all participants revealed their words
        if set(revealed_words.keys()) != set(transaction.commitments.keys()):
            print(f"❌ INCOMPLETE REVEALS: Expected {list(transaction.commitments.keys())}, got {list(revealed_words.keys())}")
            return False
        
        # Validate each commitment
        all_valid = True
        validation_results = {}
        
        for agent_id, revealed_word in revealed_words.items():
            commitment = transaction.commitments[agent_id]
            
            # Create temporary agent to validate (in production, this would be more sophisticated)
            temp_agent = CryptoCommitmentAgent("validator", self.r)
            is_valid = temp_agent.validate_other_commitment(commitment, revealed_word)
            
            validation_results[agent_id] = is_valid
            if not is_valid:
                all_valid = False
        
        # Record transaction completion
        transaction.completion_time = time.time()
        transaction.success = all_valid
        transaction.validation_proof = hashlib.sha256(
            json.dumps(validation_results, sort_keys=True).encode()
        ).hexdigest()
        
        # Update Redis
        self.r.hset(
            f"crypto_escrow:transactions:{transaction_id}",
            mapping={
                'completion_time': str(transaction.completion_time),
                'success': str(all_valid),
                'validation_proof': transaction.validation_proof,
                'validation_results': json.dumps(validation_results)
            }
        )
        
        if all_valid:
            print(f"✅ TRANSACTION VALIDATED: {transaction_id}")
            print(f"   All {len(revealed_words)} commitments verified")
            print(f"   Proof: {transaction.validation_proof[:16]}...")
        else:
            print(f"❌ TRANSACTION FAILED: {transaction_id}")
            print(f"   Invalid commitments detected")
        
        return all_valid
    
    def get_transaction_status(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """Get current transaction status"""
        
        if transaction_id not in self.active_transactions:
            return None
        
        transaction = self.active_transactions[transaction_id]
        
        return {
            'transaction_id': transaction_id,
            'task_id': transaction.task_id,
            'participants': transaction.participants,
            'commitments_received': len(transaction.commitments),
            'start_time': transaction.start_time,
            'completion_time': transaction.completion_time,
            'success': transaction.success,
            'validation_proof': transaction.validation_proof
        }

class GovernedCoordinationDemo:
    """Demonstrate governed neural coordination with crypto commitments"""
    
    def __init__(self):
        self.r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.escrow = CryptoEscrowAgent(self.r)
        
        # Create demo agents
        self.agents = {
            'alice': CryptoCommitmentAgent('alice', self.r),
            'bob': CryptoCommitmentAgent('bob', self.r),
            'charlie': CryptoCommitmentAgent('charlie', self.r)
        }
        
        print("🎯 GOVERNED COORDINATION DEMO INITIALIZED")
        print(f"   Agents: {list(self.agents.keys())}")
    
    def run_governed_coordination_demo(self):
        """Run complete governed coordination demonstration"""
        
        print("\n🚀 RUNNING GOVERNED COORDINATION DEMO")
        print("=" * 60)
        
        task_id = "demo_secure_task_001"
        participants = list(self.agents.keys())
        
        # Step 1: Start transaction
        transaction_id = self.escrow.start_transaction(task_id, participants)
        
        # Step 2: Each agent creates commitment
        commitments = {}
        for agent_id, agent in self.agents.items():
            assignment_data = {
                'role': f'role_{agent_id}',
                'effort': random.uniform(2.0, 8.0),
                'priority': random.uniform(0.5, 1.0)
            }
            
            commitment = agent.create_commitment(task_id, assignment_data)
            commitments[agent_id] = commitment
            
            # Submit to escrow
            self.escrow.receive_commitment(transaction_id, commitment)
        
        print(f"\n📋 ALL COMMITMENTS SUBMITTED")
        
        # Step 3: Simulate task execution (agents would work here)
        print(f"\n⏳ SIMULATING TASK EXECUTION...")
        time.sleep(1)  # Simulate work
        
        # Step 4: Agents reveal their commitment words
        revealed_words = {}
        for agent_id, agent in self.agents.items():
            commitment = commitments[agent_id]
            revealed_word = agent.reveal_commitment(commitment)
            revealed_words[agent_id] = revealed_word
        
        print(f"\n🔓 ALL WORDS REVEALED")
        
        # Step 5: Escrow validates transaction
        validation_result = self.escrow.validate_transaction_completion(
            transaction_id, revealed_words
        )
        
        # Step 6: Show results
        status = self.escrow.get_transaction_status(transaction_id)
        
        print(f"\n🎯 GOVERNED COORDINATION COMPLETE")
        print(f"   Transaction: {transaction_id}")
        print(f"   Success: {validation_result}")
        print(f"   Participants: {status['participants']}")
        print(f"   Validation proof: {status['validation_proof'][:16]}...")
        
        return {
            'transaction_id': transaction_id,
            'success': validation_result,
            'commitments': len(commitments),
            'validation_proof': status['validation_proof']
        }

if __name__ == "__main__":
    demo = GovernedCoordinationDemo()
    demo.run_governed_coordination_demo()