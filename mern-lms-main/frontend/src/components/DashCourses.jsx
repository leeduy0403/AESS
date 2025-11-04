import { useEffect, useState } from "react";
import { useSelector } from "react-redux";
import { Modal, Table } from "flowbite-react";
import { HiOutlineExclamationCircle } from "react-icons/hi";
import { Button } from "@mui/material";

export default function DashCourses() {
  const { currentUser } = useSelector((state) => state.user);
  const [courses, setCourses] = useState([]);
  const [showMore, setShowMore] = useState(true);
  const [showModal, setShowModal] = useState(false);
  const [courseIdToDelete, setCourseIdToDelete] = useState("");

  useEffect(() => {
    const fetchCourses = async () => {
      try {
        const res = await fetch("/api/course/get");
        const data = await res.json();
        if (res.ok) {
          setCourses(data);
          if (data?.length < 9) {
            setShowMore(false);
          }
        }
      } catch (error) {
        console.log(error.message);
      }
    };
    if (currentUser?.isAdmin) {
      fetchCourses();
    }
  }, [currentUser?._id, currentUser?.isAdmin]);

  const handleShowMore = async () => {
    const startIndex = courses?.length;
    try {
      const res = await fetch(`/api/course/get?startIndex=${startIndex}`);
      const data = await res.json();
      if (res.ok) {
        setCourses((prev) => [...prev, ...data]);
        if (data.length < 9) {
          setShowMore(false);
        }
      }
    } catch (error) {
      console.log(error.message);
    }
  };

  const handleDeleteCourse = async () => {
    setShowModal(false);
    try {
      const res = await fetch(`/api/course/delete/${courseIdToDelete}`, {
        method: "DELETE",
      });
      const data = await res.json();
      if (!res.ok) {
        console.log(data.message);
      } else {
        setCourses((prev) =>
          prev.filter((course) => course?._id !== courseIdToDelete)
        );
      }
    } catch (error) {
      console.log(error.message);
    }
  };

  return (
    <div className="table-auto overflow-x-scroll md:mx-auto p-3 scrollbar scrollbar-track-slate-100 scrollbar-thumb-slate-300 dark:scrollbar-track-slate-700 dark:scrollbar-thumb-slate-500 lg:w-3/4">
      <h1 className="my-7 text-center font-semibold text-3xl">Courses</h1>
      {currentUser.isAdmin && courses.length > 0 ? (
        <>
          <Table hoverable className="shadow-md">
            <Table.Head>
              <Table.HeadCell>Date updated</Table.HeadCell>
              <Table.HeadCell>Semester</Table.HeadCell>
              <Table.HeadCell>Academic Year</Table.HeadCell>
              <Table.HeadCell>Subject</Table.HeadCell>
              <Table.HeadCell>Faculty</Table.HeadCell>
              <Table.HeadCell>Delete</Table.HeadCell>
              {/* <Table.HeadCell>Edit</Table.HeadCell> */}
            </Table.Head>
            {courses?.length > 0 &&
              courses.map((course) => (
                <Table.Body className="divide-y" key={course?._id}>
                  <Table.Row className="bg-white dark:border-gray-700 dark:bg-gray-800">
                    <Table.Cell>
                      {new Date(course?.updatedAt).toLocaleDateString()}
                    </Table.Cell>
                    <Table.Cell className="font-medium text-gray-900 dark:text-white">
                      {course?.semester}
                    </Table.Cell>
                    <Table.Cell className="font-medium text-gray-900 dark:text-white">
                      {course?.startAcademicYear} - {course?.endAcademicYear}
                    </Table.Cell>
                    <Table.Cell className="text-gray-900 dark:text-white">
                      {course?.subjectId?.name}
                    </Table.Cell>
                    <Table.Cell className="text-gray-900 dark:text-white">
                      {course?.subjectId?.facultyId?.name}
                    </Table.Cell>
                    <Table.Cell>
                      <span
                        onClick={() => {
                          setShowModal(true);
                          setCourseIdToDelete(course?._id);
                        }}
                        className="font-medium text-red-500 hover:underline cursor-pointer"
                      >
                        Delete
                      </span>
                    </Table.Cell>
                    {/* <Table.Cell>
                      <Link
                        className="text-teal-500 hover:underline"
                        to={`/update-course/${course?._id}`}
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
        <p>You have no courses yet!</p>
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
              Are you sure you want to delete this course?
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
                onClick={handleDeleteCourse}
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
