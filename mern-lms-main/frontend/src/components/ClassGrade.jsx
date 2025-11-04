import { useEffect, useState } from "react";
import { useSelector } from "react-redux";
import { Link, useParams } from "react-router-dom";
import { Textarea } from "flowbite-react";
import { Alert, Button } from "@mui/material";
import {
  Person as PersonIcon,
  Groups as GroupsIcon,
  Done as DoneIcon,
  Close as CloseIcon,
} from "@mui/icons-material";

export default function ClassGrade() {
  const { classId } = useParams();
  const { currentUser } = useSelector((state) => state.user);
  const [assignmentsUser, setAssignmentsUser] = useState([]);
  const [assignmentsUsers, setAssignmentsUsers] = useState([]);
  const [requestData, setRequestData] = useState({});
  const [sendRequestError, setSendRequestError] = useState(false);
  const [sendRequestSuccess, setSendRequestSuccess] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchAssignments = async () => {
      try {
        let res;
        if (currentUser?.isStudent) {
          res = await fetch(
            `/api/assignment/get-last-submission-user/${classId}/${currentUser._id}`
          );
          const data = await res.json();
          if (res.ok) {
            setAssignmentsUser(data);
          }
        } else if (currentUser?.isEducator) {
          res = await fetch(
            `/api/assignment/get-last-submission-users/${classId}`
          );
          const data = await res.json();
          if (res.ok) {
            setAssignmentsUsers(data);
          }
        }
      } catch (error) {
        console.log(error.message);
      }
    };
    fetchAssignments();
  }, [
    classId,
    currentUser?._id,
    currentUser?.isStudent,
    currentUser?.isEducator,
  ]);

  const handleChangeRequestData = (e) => {
    setRequestData((prev) => {
      return {
        ...prev,
        [e.target.id]: e.target.value,
      };
    });
  };

  const handleSubmitSendRequest = async (e) => {
    e.preventDefault();
    setLoading(true);
    setSendRequestError(null);
    setSendRequestSuccess(null);
    try {
      const filteredRequestData = Object.fromEntries(
        Object.entries(requestData).filter(([_, value]) => value.trim() !== "")
      );
      const res = await fetch(`/api/question/create-from-submission`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          requestData: filteredRequestData,
          userId: currentUser?._id,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        setSendRequestError(data.message);
      } else {
        setSendRequestSuccess("Sending request(s) successfully!");
        setRequestData({});
      }
      setLoading(false);
    } catch (error) {
      setSendRequestError(error.message);
      setLoading(false);
    }
  };

  return (
    <div className="">
      {sendRequestSuccess && (
        <Alert severity="success" className="mb-4 border border-green-600">
          {sendRequestSuccess}
        </Alert>
      )}
      {sendRequestError && (
        <Alert severity="error" className="mb-4 border border-red-600">
          {sendRequestError}
        </Alert>
      )}
      {currentUser?.isStudent ? (
        <div className="overflow-x-auto rounded-lg">
          <table className="w-full text-sm text-left bg-white border border-gray-300">
            <thead className="text-white bg-[#26597C] text-center">
              <tr>
                <th className="p-4 border">Assignment</th>
                <th className="p-4 border">Grade Component</th>
                <th className="p-4 border">Total Grade</th>
                <th className="p-4 border">Graded File</th>
                <th className="p-4 border">Send Request</th>
              </tr>
            </thead>
            <tbody>
              {assignmentsUser?.length > 0 &&
                assignmentsUser.map((item, i) => (
                  <tr
                    key={i}
                    className={`transition-all ${
                      i % 2 === 0 ? "bg-white" : "bg-[#F8F8D5]"
                    }`}
                  >
                    <td className="p-4 text-center border border-gray-300 font-medium text-sky-700 hover:underline">
                      <Link
                        to={`/class/${classId}/view-attempts?assignmentId=${item?.assignment?._id}`}
                      >
                        {item?.assignment?.title}
                      </Link>
                    </td>
                    <td className="p-4 border border-gray-300">
                      {item?.lastSubmission?.scoreComponent?.length > 0 &&
                        item?.lastSubmission?.scoreComponent.map((e, j) => (
                          <div
                            key={j}
                            className="flex items-center justify-between gap-2 mb-1"
                          >
                            <div>
                              (Coef: x
                              {item?.lastSubmission?.coefficients?.[j] || 1}){" "}
                              {e}
                            </div>
                            <div className="font-semibold">
                              {item?.assignment?.type === "Individual"
                                ? item?.lastSubmission?.individualScores?.[j]
                                : item?.lastSubmission?.individualScores?.[
                                    currentUser._id
                                  ]?.[j]}
                            </div>
                          </div>
                        ))}
                    </td>
                    <td className="p-4 text-center border border-gray-300 font-semibold">
                      {item?.assignment?.isScorePublish &&
                        (item?.assignment?.type === "Individual"
                          ? item?.lastSubmission?.individualScores
                              ?.slice(0, -1)
                              ?.reduce(
                                (sum, score, idx) =>
                                  sum +
                                  score *
                                    (item?.lastSubmission?.coefficients?.[
                                      idx
                                    ] || 1),
                                0
                              )
                          : item?.lastSubmission?.individualScores?.[
                              currentUser?._id
                            ]
                              ?.slice(0, -1)
                              ?.reduce(
                                (sum, score, idx) =>
                                  sum +
                                  score *
                                    (item?.lastSubmission?.coefficients?.[
                                      idx
                                    ] || 1),
                                0
                              ))}
                    </td>
                    <td className="p-4 text-center border border-gray-300 text-sky-600">
                      {item?.lastSubmission?.nameFiles?.length > 0 &&
                        item?.lastSubmission?.nameFiles.map((url, j) => (
                          <div key={j}>
                            <Link
                              to={url}
                              target="_blank"
                              className="hover:underline"
                            >
                              {url}
                            </Link>
                          </div>
                        ))}
                    </td>
                    <td className="p-4 border border-gray-300">
                      {item?.lastSubmission?._id && (
                        <Textarea
                          id={item?.lastSubmission?._id}
                          rows={3}
                          maxLength="300"
                          value={requestData?.[item?.lastSubmission?._id] || ""}
                          placeholder="Request for regrading"
                          onChange={handleChangeRequestData}
                          className="w-full"
                        />
                      )}
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
          <div className="mt-6 flex justify-end">
            <Button
              variant="contained"
              style={{
                backgroundColor: "#26597C",
                color: "#ffffff",
                textTransform: "none",
              }}
              size="large"
              onClick={handleSubmitSendRequest}
              disabled={loading}
            >
              Send Request
            </Button>
          </div>
        </div>
      ) : (
        <div className="overflow-x-auto shadow-md rounded-lg">
          <table className="w-full text-sm text-left bg-white border border-gray-300">
            <thead className="text-white bg-[#26597C]">
              <tr className="text-center">
                <th className="p-4 border">Assignment</th>
                <th className="p-4 border">Type</th>
                <th className="p-4 border">Average Score</th>
                <th className="p-4 border">Submissions</th>
                <th className="p-4 border">Due Date</th>
                <th className="p-4 border">Published</th>
              </tr>
            </thead>
            <tbody>
              {assignmentsUsers?.length > 0 &&
                assignmentsUsers.map((item, i) => (
                  <tr
                    key={i}
                    className={`text-center transition-all ${
                      i % 2 === 0 ? "bg-white" : "bg-[#F8F8D5]"
                    }`}
                  >
                    <td className="p-4 border border-gray-300 font-medium text-sky-700 hover:underline">
                      <Link
                        to={`/class/${classId}/view-submissions?assignmentId=${item?.assignment?._id}`}
                      >
                        {item?.assignment?.title}
                      </Link>
                    </td>
                    <td className="p-4 border border-gray-300">
                      {item?.assignment?.type === "Individual" ? (
                        <PersonIcon />
                      ) : (
                        <GroupsIcon />
                      )}
                    </td>
                    <td className="p-4 border border-gray-300 font-semibold">
                      {item?.averageScore || "-"}
                    </td>
                    <td className="p-4 border border-gray-300">
                      {item?.assignment?.submissions?.length}
                    </td>
                    <td className="p-4 border border-gray-300">
                      {new Date(item?.assignment?.endDate).toLocaleString()}
                    </td>
                    <td className="p-4 border border-gray-300">
                      {item?.assignment?.isScorePublish ? (
                        <DoneIcon color="success" />
                      ) : (
                        <CloseIcon color="error" />
                      )}
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
