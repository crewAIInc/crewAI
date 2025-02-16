'use client';
import { useStore } from '@/lib/store';
import Link from 'next/link';
import { useParams } from 'next/navigation';
import { useEffect } from 'react';

interface CrewDetailProps { }

const CrewDetail = ({ }: CrewDetailProps) => {
    const params = useParams();
    const crewId = params && params.id ? String(params.id) : null;

    // Use Zustand store for state management
    const { crews, error: storeError, isLoading, fetchCrew } = useStore();

    // Find the crew from the store
    const crew = crews.find(c => c.id === crewId);
    const error = storeError?.message || null;

    useEffect(() => {
        if (crewId) {
            fetchCrew(crewId);
        }
    }, [crewId, fetchCrew]);

    if (isLoading) {
        return <div role="status" aria-live="polite">Loading crew details...</div>;
    }

    if (error) {
        return (
            <div role="alert" className="text-red-500">
                Error: {error}
            </div>
        );
    }

    if (!crew) {
        return <div role="alert">Crew not found.</div>;
    }

    return (
        <div className="p-4">
            <h1 className="text-2xl font-bold">{crew.name}</h1>
            <p className="text-gray-600">{crew.status}</p>
            {crew.description && (
                <p className="mt-2 text-gray-700">{crew.description}</p>
            )}

            <h2 className="text-xl font-semibold mt-6">Agents</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {crew.agents.length > 0 ? (
                    crew.agents.map((agent) => (
                        <div
                            key={agent.id}
                            className="bg-white rounded-lg shadow-md p-4 border border-gray-200"
                            role="article"
                            aria-labelledby={`agent-${agent.id}-name`}
                        >
                            <Link href={`/agents/${agent.id}`}>
                                <h3 id={`agent-${agent.id}-name`} className="text-lg font-medium text-gray-900">
                                    {agent.name}
                                </h3>
                            </Link>
                            <p className="text-sm text-gray-500">{agent.role}</p>

                            {agent.agent_type === 'autogen' && agent.autogen_config && (
                                <div className="mt-2">
                                    <p className="text-sm font-medium">
                                        Model: {agent.autogen_config.llm_config?.model || 'Not specified'}
                                    </p>
                                </div>
                            )}
                        </div>
                    ))
                ) : (
                    <p>No agents found for this crew.</p>
                )}
            </div>

            <h2 className="text-xl font-semibold mt-6">Tasks</h2>
            <p>Task Count: {crew.tasks.length}</p>
        </div>
    );
};

export default CrewDetail; 