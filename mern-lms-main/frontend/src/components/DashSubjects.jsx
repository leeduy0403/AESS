import { useEffect, useState } from "react";
import { useSelector } from "react-redux";
import { Modal, Table } from "flowbite-react";
import { HiOutlineExclamationCircle } from "react-icons/hi";
import { Button } from "@mui/material";

export default function DashSubjects() {
  const { currentUser } = useSelector((state) => state.user);
  const [subjects, setSubjects] = useState([]);
  const [showMore, setShowMore] = useState(true);
  const [showModal, setShowModal] = useState(false);
  const [subjectIdToDelete, setSubjectIdToDelete] = useState("");

  useEffect(() => {
    const fetchSubjects = async () => {
      try {
        const res = await fetch("/api/subject/get");
        const data = await res.json();
        if (res.ok) {
          setSubjects(data);
          if (data?.length < 9) {
            setShowMore(false);
          }
        }
      } catch (error) {
        console.log(error.message);
      }
    };
    if (currentUser?.isAdmin) {
      fetchSubjects();
    }
  }, [currentUser?._id, currentUser?.isAdmin]);

  const handleShowMore = async () => {
    const startIndex = subjects?.length;
    try {
      const res = await fetch(`/api/subject/get?startIndex=${startIndex}`);
      const data = await res.json();
      if (res.ok) {
        setSubjects((prev) => [...prev, ...data]);
        if (data.length < 9) {
          setShowMore(false);
        }
      }
    } catch (error) {
      console.log(error?.message);
    }
  };

  const handleDeleteSubject = async () => {
    setShowModal(false);
    try {
      const res = await fetch(`/api/subject/delete/${subjectIdToDelete}`, {
        method: "DELETE",
      });
      const data = await res.json();
      if (!res.ok) {
        console.log(data.message);
      } else {
        setSubjects((prev) =>
          prev.filter((subject) => subject?._id !== subjectIdToDelete)
        );
      }
    } catch (error) {
      console.log(error.message);
    }
  };
  return (
    <div className="table-auto overflow-x-scroll md:mx-auto p-3 scrollbar scrollbar-track-slate-100 scrollbar-thumb-slate-300 dark:scrollbar-track-slate-700 dark:scrollbar-thumb-slate-500 lg:w-3/4">
      <h1 className="my-7 text-center font-semibold text-3xl">Subjects</h1>
      {currentUser?.isAdmin && subjects?.length > 0 ? (
        <>
          <Table hoverable className="shadow-md">
            <Table.Head>
              <Table.HeadCell>Date updated</Table.HeadCell>
              <Table.HeadCell>Subject</Table.HeadCell>
              <Table.HeadCell>Code</Table.HeadCell>
              <Table.HeadCell>Faculty</Table.HeadCell>
              <Table.HeadCell>Delete</Table.HeadCell>
              {/* <Table.HeadCell>Edit</Table.HeadCell> */}
            </Table.Head>
            {subjects?.length > 0 &&
              subjects.map((subject) => (
                <Table.Body className="divide-y" key={subject?._id}>
                  <Table.Row className="bg-white dark:border-gray-700 dark:bg-gray-800">
                    <Table.Cell>
                      {new Date(subject?.updatedAt).toLocaleDateString()}
                    </Table.Cell>
                    <Table.Cell className="font-medium text-gray-900 dark:text-white">
                      {subject?.name}
                    </Table.Cell>
                    <Table.Cell className="text-gray-900 dark:text-white">
                      {subject?.code}
                    </Table.Cell>
                    <Table.Cell className="text-gray-900 dark:text-white">
                      {subject?.facultyId?.name}
                    </Table.Cell>
                    <Table.Cell>
                      <span
                        onClick={() => {
                          setShowModal(true);
                          setSubjectIdToDelete(subject?._id);
                        }}
                        className="font-medium text-red-500 hover:underline cursor-pointer"
                      >
                        Delete
                      </span>
                    </Table.Cell>
                    {/* <Table.Cell>
                      <Link
                        className="text-teal-500 hover:underline"
                        to={`/update-subject/${subject?._id}`}
                      >
                        <span>Edit</span>
                      </Link>
                    </Table.Cell> */}
                  </Table.Row>
                </Table.Body>
              ))}
          </Table>
          {showMore && (
            <button
              onClick={handleShowMore}
              className="w-full text-teal-500 self-center text-sm py-7"
            >
              Show more
            </button>
          )}
        </>
      ) : (
        <p>You have no subjects yet!</p>
      )}
      <Modal
        show={showModal}
        onClose={() => setShowModal(false)}
        popup
        size="md"
      >
        <Modal.Header />
        <Modal.Body>
          <div className="text-center">
            <HiOutlineExclamationCircle className="h-14 w-14 text-red-600 dark:text-gray-200 mb-4 mx-auto" />
            <h3 className="mb-5 text-lg">
              Are you sure you want to delete this subject?
            </h3>
            <div className="flex justify-center gap-4">
              <Button
                variant="contained"
                component="label"
                style={{
                  backgroundColor: "oklch(0.577 0.245 27.325)",
                  color: "#ffffff",
                  textTransform: "none",
                }}
                onClick={handleDeleteSubject}
                fullWidth
              >
                Yes, Iam sure
              </Button>
              <Button
                variant="contained"
                component="label"
                style={{
                  color: "#ffffff",
                  textTransform: "none",
                }}
                onClick={() => setShowModal(false)}
                fullWidth
              >
                No, cancel
              </Button>
            </div>
          </div>
        </Modal.Body>
      </Modal>
    </div>
  );
}
