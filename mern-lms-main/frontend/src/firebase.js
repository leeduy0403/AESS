// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
  authDomain: "blog-mern-6e704.firebaseapp.com",
  projectId: "blog-mern-6e704",
  storageBucket: "blog-mern-6e704.appspot.com",
  messagingSenderId: "724279150881",
  appId: "1:724279150881:web:b321f8a626c00f5c8b1660"
};

// Initialize Firebase
export const app = initializeApp(firebaseConfig);