// import { useEffect } from 'react';
// import { useRouter } from 'next/router';
// import { useAuthStore } from '../lib/store';

// export const ProtectedRoute = ({ children }) => {
//   const { isAuthenticated, isLoading } = useAuthStore();
//   const router = useRouter();

//   useEffect(() => {
//     if (!isLoading && !isAuthenticated) {
//       router.push('/login');
//     }
//   }, [isLoading, isAuthenticated]);

//   if (isLoading) {
//     return <div>Loading...</div>;
//   }

//   return isAuthenticated ? children : null;
// }; 