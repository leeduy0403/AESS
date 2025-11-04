import { BrowserRouter, Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import Dashboard from "./pages/Dashboard";
import SignIn from "./pages/SignIn";
import SignUp from "./pages/SignUp";
import Header from "./components/Header";
import Footer from "./components/Footer";
import PrivateRoute from "./components/PrivateRoute";
import OnlyAdminPrivateRoute from "./components/OnlyAdminPrivateRoute";
import ClassPage from "./pages/ClassPage";
import Classes from "./pages/Classes";
import ScrollToTop from "./components/ScrollToTop";
import AddSectionItem from "./pages/AddSectionItem";
import AddSection from "./pages/AddSection";
import EditSection from "./pages/EditSection";
import EditSectionItem from "./pages/EditSectionItem";
import AddAssignment from "./pages/AddAssignment";
import EditAssignment from "./pages/EditAssignment";
import AddAssignmentItem from "./pages/AddAssignmentItem";
import EditAssignmentItem from "./pages/EditAssignmentItem";
import ViewSubmissions from "./pages/ViewSubmissions";
import ViewAttempts from "./pages/ViewAttempts";
import AddSubmission from "./pages/AddSubmission";
import EditSubmission from "./pages/EditSubmission";
import AddGroup from "./pages/AddGroup";
import ViewScoreSpectrum from "./pages/ViewScoreSpectrum";
import ViewQuestions from "./pages/ViewQuestions";
import Chat from "./pages/Chat";
import Calendar from "./pages/Calendar";

export default function App() {
  return (
    <div className="flex flex-col min-h-screen">
      <BrowserRouter>
        <ScrollToTop />
        <Header />
        <main className="flex-grow">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/sign-in" element={<SignIn />} />
            <Route path="/sign-up" element={<SignUp />} />
            <Route element={<PrivateRoute />}>
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/classes" element={<Classes />} />
              <Route path="/class/:classId" element={<ClassPage />} />
              <Route path="/chat" element={<Chat />} />
              <Route path="/calendar" element={<Calendar />} />

              <Route
                path="/class/:classId/add-section"
                element={<AddSection />}
              />
              <Route
                path="/class/:classId/edit-section/:sectionId"
                element={<EditSection />}
              />
              <Route
                path="/class/:classId/add-section-item"
                element={<AddSectionItem />}
              />
              <Route
                path="/class/:classId/edit-section-item/:materialId"
                element={<EditSectionItem />}
              />

              <Route
                path="/class/:classId/add-assignment"
                element={<AddAssignment />}
              />
              <Route
                path="/class/:classId/edit-assignment/:assignmentId"
                element={<EditAssignment />}
              />
              <Route
                path="/class/:classId/add-assignment-item"
                element={<AddAssignmentItem />}
              />
              <Route
                path="/class/:classId/edit-assignment-item/:materialId"
                element={<EditAssignmentItem />}
              />

              <Route
                path="/class/:classId/view-submissions"
                element={<ViewSubmissions />}
              />
              <Route
                path="/class/:classId/view-attempts"
                element={<ViewAttempts />}
              />
              <Route
                path="/class/:classId/view-questions"
                element={<ViewQuestions />}
              />
              <Route
                path="/class/:classId/add-submission"
                element={<AddSubmission />}
              />
              <Route
                path="/class/:classId/edit-submission/:submissionId"
                element={<EditSubmission />}
              />

              <Route path="/class/:classId/add-group" element={<AddGroup />} />

              <Route
                path="/class/:classId/view-score-spectrum"
                element={<ViewScoreSpectrum />}
              />
            </Route>
            <Route element={<OnlyAdminPrivateRoute />}>
              <Route path="/sign-up" element={<SignUp />} />
            </Route>
          </Routes>
        </main>
        <Footer />
      </BrowserRouter>
    </div>
  );
}
