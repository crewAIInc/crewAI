import { useStore } from '@/lib/store';
import { Crew, CrewStatus } from '@/lib/types';
import Link from 'next/link';
import { useEffect } from 'react';

// Status badge component for better reusability
const StatusBadge = ({ status }: { status: CrewStatus }) => {
    const getStatusColor = (status: CrewStatus) => {
        switch (status) {
            case 'active':
                return 'bg-green-100 text-green-800';
            case 'running':
                return 'bg-blue-100 text-blue-800';
            case 'completed':
                return 'bg-gray-100 text-gray-800';
            case 'failed':
                return 'bg-red-100 text-red-800';
            default:
                return 'bg-gray-100 text-gray-800';
        }
    };

    return (
        <span
            className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(
                status
            )}`}
            role="status"
        >
            {status}
        </span>
    );
};

// Crew card component for better organization
const CrewCard = ({ crew }: { crew: Crew }) => (
    <Link
        href={`/crews/${crew.id}`}
        className="block p-6 bg-white rounded-lg shadow-sm hover:shadow-md transition-shadow duration-200 border border-gray-200"
        aria-label={`View details for crew ${crew.name}`}
    >
        <div className="flex justify-between items-start mb-4">
            <h3 className="text-lg font-semibold text-gray-900">{crew.name}</h3>
            <StatusBadge status={crew.status} />
        </div>
        {crew.description && (
            <p className="text-gray-600 mb-4 line-clamp-2">{crew.description}</p>
        )}
        <div className="flex justify-between items-center">
            <div className="flex items-center space-x-2">
                <span className="text-sm text-gray-500">
                    {crew.agents.length} Agent{crew.agents.length !== 1 && 's'}
                </span>
                <span className="text-gray-300">•</span>
                <span className="text-sm text-gray-500">
                    {crew.tasks.length} Task{crew.tasks.length !== 1 && 's'}
                </span>
            </div>
            <span className="text-sm text-gray-500">
                Created {new Date(crew.created_at).toLocaleDateString()}
            </span>
        </div>
    </Link>
);

// Empty state component
const EmptyState = () => (
    <div
        className="text-center py-12 px-4"
        role="status"
        aria-label="No crews found"
    >
        <svg
            className="mx-auto h-12 w-12 text-gray-400"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            aria-hidden="true"
        >
            <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"
            />
        </svg>
        <h3 className="mt-2 text-sm font-medium text-gray-900">No crews</h3>
        <p className="mt-1 text-sm text-gray-500">
            Get started by creating a new crew.
        </p>
        <div className="mt-6">
            <Link
                href="/crews/new"
                className="inline-flex items-center px-4 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                role="button"
            >
                <svg
                    className="-ml-1 mr-2 h-5 w-5"
                    xmlns="http://www.w3.org/2000/svg"
                    viewBox="0 0 20 20"
                    fill="currentColor"
                    aria-hidden="true"
                >
                    <path
                        fillRule="evenodd"
                        d="M10 3a1 1 0 011 1v5h5a1 1 0 110 2h-5v5a1 1 0 11-2 0v-5H4a1 1 0 110-2h5V4a1 1 0 011-1z"
                        clipRule="evenodd"
                    />
                </svg>
                New Crew
            </Link>
        </div>
    </div>
);

// Loading state component
const LoadingState = () => (
    <div
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 animate-pulse"
        role="status"
        aria-label="Loading crews"
    >
        {[...Array(6)].map((_, i) => (
            <div
                key={i}
                className="p-6 bg-white rounded-lg shadow-sm border border-gray-200"
            >
                <div className="flex justify-between items-start mb-4">
                    <div className="h-6 bg-gray-200 rounded w-1/3"></div>
                    <div className="h-5 bg-gray-200 rounded-full w-20"></div>
                </div>
                <div className="h-4 bg-gray-200 rounded w-3/4 mb-4"></div>
                <div className="flex justify-between items-center">
                    <div className="h-4 bg-gray-200 rounded w-1/4"></div>
                    <div className="h-4 bg-gray-200 rounded w-1/3"></div>
                </div>
            </div>
        ))}
    </div>
);

// Error state component
const ErrorState = ({ error }: { error: Error }) => (
    <div
        className="rounded-md bg-red-50 p-4"
        role="alert"
        aria-label="Error loading crews"
    >
        <div className="flex">
            <div className="flex-shrink-0">
                <svg
                    className="h-5 w-5 text-red-400"
                    xmlns="http://www.w3.org/2000/svg"
                    viewBox="0 0 20 20"
                    fill="currentColor"
                    aria-hidden="true"
                >
                    <path
                        fillRule="evenodd"
                        d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                        clipRule="evenodd"
                    />
                </svg>
            </div>
            <div className="ml-3">
                <h3 className="text-sm font-medium text-red-800">
                    Error loading crews
                </h3>
                <div className="mt-2 text-sm text-red-700">
                    <p>{error.message}</p>
                </div>
            </div>
        </div>
    </div>
);

export const CrewList = () => {
    const { crews, isLoading, error, fetchCrews } = useStore();

    useEffect(() => {
        fetchCrews();
    }, [fetchCrews]);

    if (isLoading) {
        return <LoadingState />;
    }

    if (error) {
        return <ErrorState error={error} />;
    }

    if (!crews.length) {
        return <EmptyState />;
    }

    return (
        <div
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
            role="list"
            aria-label="Crews"
        >
            {crews.map((crew) => (
                <CrewCard key={crew.id} crew={crew} />
            ))}
        </div>
    );
};

export default CrewList; 